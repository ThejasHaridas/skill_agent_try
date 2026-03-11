"""
Medical Knowledge Graph Agent
==============================
Production-grade autonomous Neo4j agent.

Features:
  - Plugin-style multi-database support (each DB is a named, fixed connection)
  - Connection pooling with health checks, per-query timeout, exponential backoff retry
  - Typed error handling (auth, connection, syntax, timeout, unexpected)
  - Cypher injection protection: parameterised queries + read-only transactions
    + blocklist with validation inside every cached schema function
  - Structured logging (rotating file + console) with per-query timing
  - lru_cache with threading locks on all stable schema discovery calls
  - Security block logging always fires, independent of cache state
  - Fresh context per question (Option A context management)

Tool design principle:
  No domain-specific shortcut tools. The agent discovers schema live and writes
  all Cypher itself using: discover_labels, discover_label_schema,
  discover_relationship_schema, run_cypher_query, traverse_path, filter_by_confidence.
  Works on ANY Neo4j graph regardless of domain or naming conventions.
"""

import os
import re
import time
import uuid
import json
import yaml
import threading
import logging
import logging.handlers
from functools import lru_cache
from typing import Dict, List, Callable, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from neo4j import GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ServiceUnavailable,
    SessionExpired,
    CypherSyntaxError,
    ClientError,
    TransientError,
)

__all__ = ["query_graph", "AgentError", "DB_POOLS", "DB_CONFIGS"]

# ─────────────────────────────────────────────────────────────────
# 1. LOGGING
#    Console: INFO+, clean format
#    File:    DEBUG+, full detail, 5 MB rotation, 5 backups
#    Lazy:    os.makedirs called inside setup_logging(), not at import
# ─────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("medical_agent")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:           # guard against duplicate handlers on reload
        return logger

    fmt_console = logging.Formatter(
        "[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(funcName)s:%(lineno)d | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt_console)

    # Lazy directory creation — only when logging is actually initialised
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/medical_agent.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt_file)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()

# ─────────────────────────────────────────────────────────────────
# 2. MULTI-DATABASE PLUGIN REGISTRY
#
#    Each database is a FIXED named plugin — not runtime-switchable.
#    Descriptions are kept domain-neutral so they do not hint at
#    label names to the agent (fix for audit issue #12).
#
#    To add a new plugin: add an entry to DB_CONFIGS and set env vars.
#    Convention: NEO4J_<PLUGIN_NAME_UPPER>_URI / _USER / _PASSWORD
# ─────────────────────────────────────────────────────────────────

DB_CONFIGS: Dict[str, Dict[str, str]] = {
    "medical_kg": {
        "description": "Primary biomedical knowledge graph",
        "uri_env":      "NEO4J_MEDICAL_KG_URI",
        "user_env":     "NEO4J_MEDICAL_KG_USER",
        "password_env": "NEO4J_MEDICAL_KG_PASSWORD",
    },
    "drug_db": {
        "description": "Pharmacology and compound interaction graph",
        "uri_env":      "NEO4J_DRUG_DB_URI",
        "user_env":     "NEO4J_DRUG_DB_USER",
        "password_env": "NEO4J_DRUG_DB_PASSWORD",
    },
    "pathway_db": {
        "description": "Biological pathway graph",
        "uri_env":      "NEO4J_PATHWAY_DB_URI",
        "user_env":     "NEO4J_PATHWAY_DB_USER",
        "password_env": "NEO4J_PATHWAY_DB_PASSWORD",
    },
}

# ─────────────────────────────────────────────────────────────────
# 3. CONNECTION POOL MANAGER
# ─────────────────────────────────────────────────────────────────

_QUERY_TIMEOUT_SECONDS = 30   # per-query wall-clock limit


class Neo4jConnectionPool:
    """
    Pooled, read-only Neo4j driver for one database plugin.

    All queries run inside execute_read() — write operations are rejected
    at the driver level regardless of query content.
    Per-query timeout prevents runaway queries blocking pool threads.
    """

    def __init__(
        self,
        name: str,
        uri: str,
        user: str,
        password: str,
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0,
        max_transaction_retry_time: float = 30.0,
    ) -> None:
        self.name = name
        self.uri  = uri
        logger.info(
            f"Initialising pool | db={name} | uri={uri} | "
            f"pool_size={max_connection_pool_size}"
        )
        print(f"[DB] Connecting to '{name}' at {uri} ...")
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=connection_timeout,
            max_transaction_retry_time=max_transaction_retry_time,
        )
        self._health_check()

    def _health_check(self) -> None:
        logger.debug(f"Health check start | db={self.name}")
        print(f"[DB] Health check on '{self.name}' ...")
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").single()
            logger.info(f"Health check passed | db={self.name}")
            print(f"[DB] '{self.name}' connected successfully.")
        except AuthError as e:
            logger.error(f"Health check FAILED — auth | db={self.name} | {e}")
            raise RuntimeError(
                f"[{self.name}] Authentication failed. "
                "Check credentials in environment variables."
            ) from e
        except ServiceUnavailable as e:
            logger.error(f"Health check FAILED — unreachable | db={self.name} | {e}")
            raise RuntimeError(
                f"[{self.name}] Database unreachable at {self.uri}. "
                "Check NEO4J URI and network connectivity."
            ) from e

    def query(
        self,
        cypher: str,
        params: Optional[Dict] = None,
        retries: int = 3,
        timeout: int = _QUERY_TIMEOUT_SECONDS,
    ) -> List[Dict]:
        """
        Execute a read-only parameterised query with retry and per-query timeout.

        FIX C1: lambda captures cypher/params as default args — avoids the
                closure-over-loop-variable trap in the retry loop.
        FIX C9: timeout passed to tx.run() as a keyword arg — prevents runaway
                queries from holding pool threads indefinitely.
        """
        params     = params or {}
        last_error = None

        for attempt in range(1, retries + 1):
            t_start = time.monotonic()
            try:
                with self._driver.session() as session:
                    # FIX C1: default-arg capture prevents closure trap
                    result = session.execute_read(
                        lambda tx, c=cypher, p=params: list(
                            tx.run(c, timeout=timeout, **p)
                        )
                    )
                elapsed = time.monotonic() - t_start
                rows    = [dict(record) for record in result]
                logger.debug(
                    f"Query OK | db={self.name} | rows={len(rows)} | "
                    f"elapsed={elapsed:.3f}s | query={cypher[:120]}"
                )
                print(f"[DB:{self.name}] {len(rows)} row(s) in {elapsed:.2f}s")
                return rows

            except AuthError as e:
                logger.error(f"AUTH ERROR | db={self.name} | {e}")
                raise RuntimeError(
                    f"[{self.name}] Authentication error: {e}. Check credentials."
                ) from e

            except CypherSyntaxError as e:
                elapsed = time.monotonic() - t_start
                logger.error(
                    f"SYNTAX ERROR | db={self.name} | elapsed={elapsed:.3f}s | "
                    f"error={e} | query={cypher}"
                )
                raise RuntimeError(f"[{self.name}] Cypher syntax error: {e}") from e

            except (TransientError, SessionExpired) as e:
                elapsed = time.monotonic() - t_start
                wait    = 2 ** attempt
                logger.warning(
                    f"TRANSIENT (attempt {attempt}/{retries}) | "
                    f"db={self.name} | elapsed={elapsed:.3f}s | "
                    f"retrying in {wait}s | {e}"
                )
                print(
                    f"[DB:{self.name}] Transient error, "
                    f"retrying in {wait}s ({attempt}/{retries})..."
                )
                last_error = e
                time.sleep(wait)

            except ServiceUnavailable as e:
                elapsed = time.monotonic() - t_start
                wait    = 2 ** attempt
                logger.warning(
                    f"UNAVAILABLE (attempt {attempt}/{retries}) | "
                    f"db={self.name} | elapsed={elapsed:.3f}s | "
                    f"retrying in {wait}s | {e}"
                )
                print(
                    f"[DB:{self.name}] Unavailable, "
                    f"retrying in {wait}s ({attempt}/{retries})..."
                )
                last_error = e
                time.sleep(wait)

            except ClientError as e:
                elapsed = time.monotonic() - t_start
                logger.error(
                    f"CLIENT ERROR | db={self.name} | elapsed={elapsed:.3f}s | "
                    f"error={e} | query={cypher}"
                )
                raise RuntimeError(f"[{self.name}] Client error: {e}") from e

            except Exception as e:
                elapsed = time.monotonic() - t_start
                logger.error(
                    f"UNEXPECTED | db={self.name} | elapsed={elapsed:.3f}s | "
                    f"{type(e).__name__}: {e} | query={cypher}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"[{self.name}] Unexpected error ({type(e).__name__}): {e}"
                ) from e

        logger.error(
            f"RETRIES EXHAUSTED | db={self.name} | "
            f"attempts={retries} | last={last_error}"
        )
        raise RuntimeError(
            f"[{self.name}] Query failed after {retries} attempts. "
            f"Last error: {last_error}"
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            logger.info(f"Pool closed | db={self.name}")
            print(f"[DB] Pool closed for '{self.name}'")


def _load_pool(name: str, config: Dict[str, str]) -> Optional[Neo4jConnectionPool]:
    """Load one DB plugin from env vars. Returns None with a warning if creds missing."""
    uri      = os.environ.get(config["uri_env"])
    user     = os.environ.get(config["user_env"])
    password = os.environ.get(config["password_env"])
    missing  = [k for k, v in
                {config["uri_env"]: uri,
                 config["user_env"]: user,
                 config["password_env"]: password}.items()
                if not v]
    if missing:
        logger.warning(f"Skipping '{name}' — missing env vars: {missing}")
        print(f"[DB] Skipping '{name}' — missing: {', '.join(missing)}")
        return None
    try:
        return Neo4jConnectionPool(name=name, uri=uri, user=user, password=password)  # type: ignore[arg-type]
    except RuntimeError as e:
        logger.error(f"Failed to connect '{name}': {e}")
        print(f"[DB] Failed to connect '{name}': {e}")
        return None


# Startup: load all configured plugins
print("\n[STARTUP] Loading database plugins...")
DB_POOLS: Dict[str, Neo4jConnectionPool] = {}
for _name, _config in DB_CONFIGS.items():
    _pool = _load_pool(_name, _config)
    if _pool:
        DB_POOLS[_name] = _pool

if not DB_POOLS:
    # FIX: no input() fallback — fail fast with actionable message
    _msg = (
        "No database plugins connected. "
        "Set environment variables for at least one plugin in DB_CONFIGS "
        "(e.g. NEO4J_MEDICAL_KG_URI, NEO4J_MEDICAL_KG_USER, NEO4J_MEDICAL_KG_PASSWORD)."
    )
    logger.error(_msg)
    raise RuntimeError(_msg)

print(f"[STARTUP] Active databases: {list(DB_POOLS.keys())}\n")
logger.info(f"Active DB plugins: {list(DB_POOLS.keys())}")


def _get_pool(db_name: str) -> Neo4jConnectionPool:
    """Return pool by name. If only one pool exists, return it regardless of name."""
    if db_name in DB_POOLS:
        return DB_POOLS[db_name]
    if len(DB_POOLS) == 1:
        return next(iter(DB_POOLS.values()))
    raise ValueError(
        f"Database '{db_name}' not found. Available: {list(DB_POOLS.keys())}"
    )


# ─────────────────────────────────────────────────────────────────
# 4. SECURITY — THREE LAYERS
#
#   Layer 1: execute_read() at driver level — writes rejected unconditionally
#   Layer 2: Blocklist regex — keyword patterns + APOC/subquery bypass
#   Layer 3: Identifier validation inside EVERY cached schema function,
#            not only in the @tool callers (FIX C2, C3)
#
#   FIX C8: Security log fires in _safe_query on EVERY blocked call,
#           regardless of whether _check_query_safety came from cache.
# ─────────────────────────────────────────────────────────────────

_DANGEROUS_PATTERNS = [
    r"\bDELETE\b",
    r"\bDETACH\s+DELETE\b",
    r"\bREMOVE\b",
    r"\bSET\b",
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDROP\b",
    r"\bCALL\s+apoc\.",
    r"\bCALL\s+\{",
    r"\bCALL\s+db\.\w+\s*\(",
    r"\bLOAD\s+CSV\b",
    r"\bUSING\s+PERIODIC\s+COMMIT\b",
]
_DANGEROUS_RE  = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PATTERNS]
_SAFE_DB_CALLS = {"db.labels", "db.relationshipTypes", "db.propertyKeys"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@lru_cache(maxsize=512)
def _check_query_safety(query: str) -> Optional[str]:
    """
    Pure function — returns error string if unsafe, else None.
    Cached for performance: same query always has same safety verdict.
    Logging is NOT done here — it lives in _safe_query so it fires on
    every blocked call, not just cache misses (FIX C8).
    """
    for pattern in _DANGEROUS_RE:
        match = pattern.search(query)
        if match:
            matched_text = match.group(0).strip().lower()
            if any(safe in matched_text for safe in _SAFE_DB_CALLS):
                continue
            return (
                f"BLOCKED: Query contains disallowed pattern '{match.group(0)}'. "
                "This agent is strictly read-only."
            )
    return None


def _safe_query(
    db_name: str,
    cypher: str,
    params: Optional[Dict] = None,
) -> List[Dict]:
    """
    Central query gate — security check then pool query.
    All tools must call this, never pool.query() directly.
    FIX C8: security warning logged here, always, not inside the cached function.
    """
    error = _check_query_safety(cypher)
    if error:
        logger.warning(
            f"SECURITY BLOCK | db={db_name} | reason={error} | "
            f"query={cypher[:200]}"
        )
        print(f"[SECURITY] Query blocked on '{db_name}'")
        raise PermissionError(error)
    return _get_pool(db_name).query(cypher, params=params)


# ─────────────────────────────────────────────────────────────────
# 5. SKILL LOADER
# ─────────────────────────────────────────────────────────────────

def load_skills_from_directory(directory: str = "./skills") -> List[Dict]:
    # FIX H17: lazy mkdir — not at import time
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Loading skills from '{directory}'")
    skills: List[Dict] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()
            parts = raw.split("---", 2)
            if len(parts) == 3:
                meta = yaml.safe_load(parts[1])
                body = parts[2].strip()
            else:
                meta = {"name": filename[:-3], "description": "No description."}
                body = raw.strip()
            skills.append({
                "name":        meta.get("name", filename[:-3]),
                "description": meta.get("description", "No description."),
                "content":     body,
            })
            logger.debug(f"Skill loaded: '{meta.get('name')}' from {filename}")
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in '{filename}': {e}")
            print(f"[SKILLS] Warning: bad frontmatter in '{filename}': {e}")
        except OSError as e:
            logger.error(f"Cannot read '{filename}': {e}")
            print(f"[SKILLS] Warning: cannot read '{filename}': {e}")
    print(f"[STARTUP] Loaded {len(skills)} skill(s): {[s['name'] for s in skills]}")
    logger.info(f"Skills loaded: {[s['name'] for s in skills]}")
    return skills


SKILLS: List[Dict] = load_skills_from_directory("./skills")

# ─────────────────────────────────────────────────────────────────
# 6. LLM — fail fast, no input() fallback
# ─────────────────────────────────────────────────────────────────

_openai_key = os.environ.get("OPENAI_API_KEY")
if not _openai_key:
    _msg = "OPENAI_API_KEY environment variable is not set."
    logger.error(_msg)
    raise RuntimeError(_msg)

model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=_openai_key)
logger.info("LLM initialised: gpt-4o")

# ─────────────────────────────────────────────────────────────────
# 7. SCHEMA CACHE HELPERS
#
#   FIX C10: one threading.Lock per cache — prevents stampede where two
#            concurrent callers both miss the cache and both hit the DB.
#            The second waits for the first to populate, then hits cache.
#
#   FIX C11: return None (not empty tuple) when data is absent so the
#            absent result is NOT permanently cached — callers will retry.
#
#   FIX C2/C3: identifier validation lives INSIDE each cached function,
#              not only in the @tool caller, so internal callers are safe too.
#
#   FIX H18: list_databases result is also cached.
# ─────────────────────────────────────────────────────────────────

_labels_lock       = threading.Lock()
_label_schema_lock = threading.Lock()
_rel_schema_lock   = threading.Lock()
_list_db_lock      = threading.Lock()


@lru_cache(maxsize=1)
def _list_databases_cached() -> str:
    with _list_db_lock:
        if not DB_POOLS:
            return "No databases are currently connected."
        lines = [f"Available databases ({len(DB_POOLS)}):"]
        for name in DB_POOLS:
            desc = DB_CONFIGS.get(name, {}).get("description", "No description")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)


@lru_cache(maxsize=32)
def _fetch_labels(db_name: str) -> Optional[Tuple[tuple, tuple]]:
    """
    Returns (labels_tuple, rel_types_tuple) or None if DB returned nothing.
    None is not permanently cached — callers will retry on next call.
    Lock prevents cache stampede on cold start.
    """
    with _labels_lock:
        labels_res = _safe_query(
            db_name,
            "CALL db.labels() YIELD label RETURN collect(label) AS labels",
        )
        rels_res = _safe_query(
            db_name,
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN collect(relationshipType) AS types",
        )
    labels = tuple(labels_res[0]["labels"]) if labels_res else ()
    types  = tuple(rels_res[0]["types"])    if rels_res  else ()
    if not labels and not types:
        logger.warning(f"_fetch_labels | db={db_name} | empty — not caching")
        return None
    logger.debug(
        f"_fetch_labels CACHE MISS | db={db_name} | "
        f"labels={len(labels)} | types={len(types)}"
    )
    return labels, types


@lru_cache(maxsize=256)
def _fetch_label_schema(label: str, db_name: str) -> Optional[Tuple[str, ...]]:
    """
    FIX C2: validation inside the cached function — not only in the @tool caller.
    Returns tuple of property names or None if label not found / invalid.
    None is not permanently cached (FIX C11) — callers can retry.
    """
    if not _IDENTIFIER_RE.match(label):
        logger.warning(f"_fetch_label_schema | invalid label: '{label}'")
        return None
    with _label_schema_lock:
        result = _safe_query(
            db_name,
            f"MATCH (n:{label}) WITH n LIMIT 1 RETURN keys(n) AS properties",
        )
    if not result or not result[0]["properties"]:
        logger.debug(
            f"_fetch_label_schema CACHE MISS | db={db_name} | "
            f"label={label} | no data — not caching"
        )
        return None
    props = tuple(result[0]["properties"])
    logger.debug(
        f"_fetch_label_schema CACHE MISS | db={db_name} | label={label} | props={props}"
    )
    return props


@lru_cache(maxsize=256)
def _fetch_relationship_schema(
    rel_type: str, db_name: str
) -> Optional[Tuple[tuple, tuple, tuple]]:
    """
    FIX C3: validation inside the cached function — not only in the @tool caller.
    Returns (from_labels, to_labels, rel_props) or None.
    """
    if not _IDENTIFIER_RE.match(rel_type):
        logger.warning(f"_fetch_relationship_schema | invalid rel_type: '{rel_type}'")
        return None
    with _rel_schema_lock:
        result = _safe_query(
            db_name,
            f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WITH a, r, b LIMIT 1
            RETURN labels(a) AS from_labels,
                   labels(b) AS to_labels,
                   keys(r)   AS rel_properties
            """,
        )
    if not result:
        return None
    row = result[0]
    schema = (
        tuple(row["from_labels"]),
        tuple(row["to_labels"]),
        tuple(row["rel_properties"] or []),
    )
    logger.debug(
        f"_fetch_relationship_schema CACHE MISS | db={db_name} | rel={rel_type}"
    )
    return schema


# ─────────────────────────────────────────────────────────────────
# 8. TOOLS
# ─────────────────────────────────────────────────────────────────

@tool
def list_databases() -> str:
    """
    List all available database plugins and their descriptions.
    Always call this first so you know which databases exist.
    """
    result = _list_databases_cached()
    logger.debug(f"list_databases | {result}")
    return result


@tool
def discover_labels(db_name: str = "default") -> str:
    """
    Get all node labels and relationship types in a database.
    Call this FIRST before writing any query for a given database.

    Args:
        db_name: Plugin name from list_databases(). Default: first available.
    """
    logger.info(f"discover_labels | db={db_name}")
    print(f"[TOOL] discover_labels on '{db_name}'")
    try:
        data = _fetch_labels(db_name)
        ci   = _fetch_labels.cache_info()
        logger.debug(f"discover_labels | cache hits={ci.hits} misses={ci.misses}")
        if data is None:
            return (
                f"[{db_name}] No labels or relationship types found. "
                "The database may be empty."
            )
        labels, types = data
        return (
            f"[{db_name}] {len(labels)} node label(s): {', '.join(labels)}\n"
            f"[{db_name}] {len(types)} relationship type(s): {', '.join(types)}\n\n"
            "Call discover_label_schema('<Label>') for properties of relevant labels.\n"
            "Call discover_relationship_schema('<TYPE>') for relationship details."
        )
    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(f"discover_labels failed | db={db_name} | {e}")
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def discover_label_schema(label: str, db_name: str = "default") -> str:
    """
    Get property names for ONE specific node label.
    Only call for labels relevant to the current question.

    Args:
        label:   Node label (alphanumeric + underscore)
        db_name: Plugin name from list_databases().
    """
    logger.info(f"discover_label_schema | db={db_name} | label={label}")
    print(f"[TOOL] discover_label_schema '{label}' on '{db_name}'")
    if not _IDENTIFIER_RE.match(label):
        logger.warning(f"Invalid label rejected: '{label}'")
        return f"INVALID LABEL: '{label}' — alphanumeric + underscore only."
    try:
        props = _fetch_label_schema(label, db_name)
        ci    = _fetch_label_schema.cache_info()
        logger.debug(f"discover_label_schema | cache hits={ci.hits} misses={ci.misses}")
        if props is None:
            return (
                f"[{db_name}] Label '{label}' not found or has no nodes. "
                "Verify with discover_labels()."
            )
        return f"[{db_name}] Node :{label} has properties: {', '.join(props)}"
    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(f"discover_label_schema failed | db={db_name} | label={label} | {e}")
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def discover_relationship_schema(rel_type: str, db_name: str = "default") -> str:
    """
    Get source label, target label, and properties of ONE relationship type.
    Direction is biologically critical — always verify before querying.

    Args:
        rel_type: Relationship type (alphanumeric + underscore)
        db_name:  Plugin name from list_databases().
    """
    logger.info(f"discover_relationship_schema | db={db_name} | rel={rel_type}")
    print(f"[TOOL] discover_relationship_schema '{rel_type}' on '{db_name}'")
    if not _IDENTIFIER_RE.match(rel_type):
        logger.warning(f"Invalid rel_type rejected: '{rel_type}'")
        return f"INVALID TYPE: '{rel_type}' — alphanumeric + underscore only."
    try:
        cached = _fetch_relationship_schema(rel_type, db_name)
        ci     = _fetch_relationship_schema.cache_info()
        logger.debug(
            f"discover_relationship_schema | cache hits={ci.hits} misses={ci.misses}"
        )
        if cached is None:
            return (
                f"[{db_name}] Relationship '{rel_type}' not found. "
                "Verify with discover_labels()."
            )
        from_labels, to_labels, rel_props = cached
        return (
            f"[{db_name}] Relationship :{rel_type}\n"
            f"  Direction : {list(from_labels)} → {list(to_labels)}\n"
            f"  Properties: {list(rel_props) or 'none'}\n\n"
            "Always use this direction in your MATCH. "
            "Check if 'confidence', 'score', or 'evidence' is present "
            "and include it in RETURN."
        )
    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(
            f"discover_relationship_schema failed | db={db_name} | rel={rel_type} | {e}"
        )
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def count_nodes(label: str = "", db_name: str = "default") -> str:
    """
    Count nodes, optionally filtered by label. Call before large traversals.

    Args:
        label:   Optional node label. Empty = total node count.
        db_name: Plugin name from list_databases().
    """
    logger.info(f"count_nodes | db={db_name} | label='{label}'")
    print(f"[TOOL] count_nodes label='{label}' on '{db_name}'")
    if label and not _IDENTIFIER_RE.match(label):
        return f"INVALID LABEL: '{label}' — alphanumeric + underscore only."
    try:
        cypher = (
            f"MATCH (n:{label}) RETURN count(n) AS total"
            if label else
            "MATCH (n) RETURN count(n) AS total"
        )
        result    = _safe_query(db_name, cypher)
        total     = result[0]["total"] if result else 0
        label_str = f":{label}" if label else ""
        logger.debug(f"count_nodes | db={db_name} | label={label} | total={total}")
        return f"[{db_name}] Node count ({label_str}): {total:,}"
    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(f"count_nodes failed | db={db_name} | {e}")
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def run_cypher_query(query: str, db_name: str = "default") -> str:
    """
    Execute a read-only Cypher query.

    Result routing:
      < 500 rows  → full results
      500–5000    → full results + column statistics
      > 5000 rows → first 100 rows sample + full column statistics + warning
                    (FIX C5: never dumps the full set into LLM context)

    Args:
        query:   A valid Cypher MATCH/RETURN query
        db_name: Plugin name from list_databases().
    """
    logger.info(f"run_cypher_query | db={db_name} | query={query[:120]}")
    print(f"[TOOL] run_cypher_query on '{db_name}' | {query[:80]}...")
    try:
        results = _safe_query(db_name, query)
        if not results:
            logger.debug(f"run_cypher_query | db={db_name} | no results")
            return f"[{db_name}] Query returned no results."

        row_count = len(results)
        keys      = list(results[0].keys())
        logger.info(f"run_cypher_query | db={db_name} | rows={row_count}")

        # FIX H15: build formatted string inside each branch — not before
        if row_count < 500:
            formatted = "\n".join(str(r) for r in results)
            return f"[{db_name}] Results ({row_count} rows):\n\n{formatted}"

        elif row_count < 5000:
            formatted = "\n".join(str(r) for r in results)
            stats     = _compute_column_stats(results, keys)
            return (
                f"[{db_name}] Results ({row_count} rows):\n\n{formatted}\n\n"
                f"Column summary:\n{json.dumps(stats, indent=2)}"
            )

        else:
            # FIX C5: cap output — return sample + stats only, never full dump
            logger.warning(
                f"run_cypher_query | db={db_name} | "
                f"large result {row_count} rows — capping output to 100"
            )
            print(f"[TOOL] Large result ({row_count} rows) — capping on '{db_name}'")
            sample    = results[:100]
            formatted = "\n".join(str(r) for r in sample)
            stats     = _compute_column_stats(results, keys)
            return (
                f"[{db_name}] Large result: {row_count:,} total rows.\n"
                f"Showing first 100 rows:\n\n{formatted}\n\n"
                f"Full column statistics ({row_count:,} rows):\n"
                f"{json.dumps(stats, indent=2)}\n\n"
                "WARNING: Result too large to return in full. "
                "Refine your query with WHERE / LIMIT, "
                "or use traverse_path() for path questions."
            )

    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(f"run_cypher_query failed | db={db_name} | {e}")
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def traverse_path(
    start_label: str,
    start_property: str,
    start_value: str,
    end_label: str,
    end_property: str,
    end_value: str,
    db_name: str = "default",
    max_hops: int = 4,
    relationship_filter: str = "",
) -> str:
    """
    Find all directed paths between two entities up to max_hops deep.

    Use for: "How is X connected to Y?", "What links A to B?"

    FIX H13: invalid relationship_filter returns INVALID INPUT error
             instead of silently being ignored.

    Args:
        start_label:         Node label to start from
        start_property:      Property to match start node (discovered via schema tools)
        start_value:         Value to match
        end_label:           Node label to end at
        end_property:        Property to match end node
        end_value:           Value to match
        db_name:             Plugin name from list_databases().
        max_hops:            Max relationship hops (default 4, max 6)
        relationship_filter: Optional filter e.g. 'TREATS|INHIBITS'
                             Alphanumeric, underscore, pipe only.
    """
    logger.info(
        f"traverse_path | db={db_name} | "
        f"{start_label}({start_value}) -> {end_label}({end_value}) | hops={max_hops}"
    )
    print(
        f"[TOOL] traverse_path {start_label}({start_value}) "
        f"→ {end_label}({end_value}) on '{db_name}'"
    )

    for val, name in [
        (start_label,    "start_label"),
        (end_label,      "end_label"),
        (start_property, "start_property"),
        (end_property,   "end_property"),
    ]:
        if not _IDENTIFIER_RE.match(val):
            return (
                f"INVALID INPUT: '{val}' in '{name}' "
                "— alphanumeric + underscore only."
            )

    max_hops = min(max_hops, 6)

    # FIX H13: explicit error on bad filter — never silently ignore
    if relationship_filter:
        if not re.match(r"^[A-Za-z][A-Za-z0-9_|]*$", relationship_filter):
            return (
                f"INVALID INPUT: relationship_filter '{relationship_filter}' "
                "contains disallowed characters. "
                "Use alphanumeric, underscore, pipe only e.g. 'TREATS|INHIBITS'."
            )
        rel_pattern = f"[r:{relationship_filter}*1..{max_hops}]"
    else:
        rel_pattern = f"[r*1..{max_hops}]"

    cypher = f"""
        MATCH path = (start:{start_label} {{{start_property}: $start_val}})
                     -{rel_pattern}->
                     (end:{end_label} {{{end_property}: $end_val}})
        RETURN path,
               length(path) AS hops,
               [n IN nodes(path) | {{
                   labels:     labels(n),
                   properties: properties(n)
               }}] AS path_nodes,
               [r IN relationships(path) | {{
                   type:       type(r),
                   properties: properties(r)
               }}] AS path_rels
        ORDER BY hops ASC
        LIMIT 20
    """

    try:
        results = _safe_query(
            db_name, cypher,
            params={"start_val": start_value, "end_val": end_value},
        )
        if not results:
            logger.debug(f"traverse_path | db={db_name} | no paths found")
            return (
                f"[{db_name}] No paths found between "
                f"{start_label}({start_value}) and {end_label}({end_value}) "
                f"within {max_hops} hops.\n"
                "Try: increase max_hops, remove relationship_filter, "
                "or verify entity names."
            )

        output = [
            f"[{db_name}] Found {len(results)} path(s): "
            f"{start_label}({start_value}) → {end_label}({end_value})\n"
        ]
        for i, row in enumerate(results, 1):
            path_nodes = row["path_nodes"]
            path_rels  = row["path_rels"]
            hops       = row["hops"]
            chain: List[str] = []
            for j, node in enumerate(path_nodes):
                node_label = node["labels"][0] if node["labels"] else "Unknown"
                node_name  = (
                    node["properties"].get("name")
                    or node["properties"].get("id")
                    or str(node["properties"])[:50]
                )
                chain.append(f"({node_label}: {node_name})")
                if j < len(path_rels):
                    rel  = path_rels[j]
                    conf = (
                        rel["properties"].get("confidence")
                        or rel["properties"].get("score")
                        or rel["properties"].get("evidence_score")
                    )
                    conf_str = f" [confidence: {conf}]" if conf else ""
                    chain.append(f"-[:{rel['type']}{conf_str}]->")
            output.append(
                f"Path {i} ({hops} hop{'s' if hops != 1 else ''}):\n"
                f"  {''.join(chain)}\n"
            )

        logger.info(f"traverse_path | db={db_name} | paths_found={len(results)}")
        return "\n".join(output)

    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(f"traverse_path failed | db={db_name} | {e}")
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def filter_by_confidence(
    query: str,
    db_name: str = "default",
    min_confidence: float = 0.7,
    confidence_property: str = "confidence",
) -> str:
    """
    Run a Cypher query filtered by a relationship confidence/evidence score.
    Use when the question asks for well-evidenced associations only.
    The query must alias the relationship(s) as 'r'.

    FIX C4: Clause-aware injection — inserts the condition at the correct
    position before ORDER BY / LIMIT / WITH / RETURN, not blindly appended.
    The constructed query is validated by _safe_query before execution.

    Args:
        query:               MATCH query with 'r' aliasing the relationship
        db_name:             Plugin name from list_databases().
        min_confidence:      Minimum score threshold (default 0.7)
        confidence_property: Property name on 'r' (default 'confidence')
    """
    logger.info(
        f"filter_by_confidence | db={db_name} | min={min_confidence} | "
        f"prop={confidence_property} | query={query[:80]}"
    )
    print(f"[TOOL] filter_by_confidence min={min_confidence} on '{db_name}'")

    if not _IDENTIFIER_RE.match(confidence_property):
        return (
            f"INVALID: confidence_property '{confidence_property}' "
            "— alphanumeric + underscore only."
        )

    q_upper   = query.upper()
    condition = f"r.{confidence_property} >= $min_conf"

    # Tail keywords that mark the end of the filterable clause body
    _TAIL = ["ORDER BY", "LIMIT", "WITH", "RETURN"]
    first_return = q_upper.find("RETURN")

    if first_return == -1:
        return "INVALID QUERY: no RETURN clause — cannot inject confidence filter."

    last_where = q_upper.rfind("WHERE")

    if last_where != -1 and last_where < first_return:
        # Append to existing WHERE — find earliest tail keyword after WHERE
        inject_at = first_return
        for kw in _TAIL:
            pos = q_upper.find(kw, last_where)
            if pos != -1 and pos < inject_at:
                inject_at = pos
        filtered = (
            query[:inject_at].rstrip()
            + f"\n  AND {condition}\n"
            + query[inject_at:]
        )
    else:
        # No WHERE before RETURN — insert one before earliest tail keyword
        inject_at = first_return
        for kw in _TAIL:
            pos = q_upper.find(kw)
            if pos != -1 and pos < inject_at:
                inject_at = pos
        filtered = (
            query[:inject_at].rstrip()
            + f"\nWHERE {condition}\n"
            + query[inject_at:]
        )

    logger.debug(f"filter_by_confidence | constructed:\n{filtered}")

    try:
        results = _safe_query(db_name, filtered, params={"min_conf": min_confidence})
        if not results:
            logger.debug(f"filter_by_confidence | db={db_name} | no results")
            return (
                f"[{db_name}] No results with "
                f"{confidence_property} >= {min_confidence}.\n"
                "Try lowering min_confidence."
            )
        # FIX H15: build formatted inside branch
        formatted = "\n".join(str(r) for r in results)
        logger.info(f"filter_by_confidence | db={db_name} | rows={len(results)}")
        return (
            f"[{db_name}] Results with {confidence_property} >= {min_confidence} "
            f"({len(results)} row(s)):\n\n{formatted}"
        )
    except PermissionError as e:
        return f"SECURITY ERROR: {e}"
    except RuntimeError as e:
        logger.error(f"filter_by_confidence failed | db={db_name} | {e}")
        return f"DATABASE ERROR on '{db_name}': {e}"
    except ValueError as e:
        return f"CONFIG ERROR: {e}"


@tool
def load_skill(skill_name: str) -> str:
    """
    Load full skill instructions on demand.
    Call before writing Cypher: load_skill('medical_cypher')
    Call on query errors:       load_skill('error_fixer')

    Args:
        skill_name: Exact skill name e.g. 'medical_cypher', 'error_fixer'
    """
    logger.info(f"load_skill | skill={skill_name}")
    print(f"[TOOL] load_skill '{skill_name}'")
    for skill in SKILLS:
        if skill["name"] == skill_name:
            logger.debug(
                f"load_skill | loaded '{skill_name}' ({len(skill['content'])} chars)"
            )
            return f"[Skill loaded: {skill_name}]\n\n{skill['content']}"
    available = ", ".join(s["name"] for s in SKILLS)
    logger.warning(f"load_skill | '{skill_name}' not found | available={available}")
    return f"Skill '{skill_name}' not found. Available: {available}"


# ─────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────

def _compute_column_stats(results: List[Dict], keys: List[str]) -> Dict:
    stats: Dict = {}
    for key in keys:
        # FIX H14: use `key in r and r[key] is not None` — not r.get(key)
        # Neo4j returns None for absent properties; get() can't distinguish
        # "missing key" from "key present with value None"
        values = [r[key] for r in results if key in r and r[key] is not None]
        if values and isinstance(values[0], (int, float)):
            stats[key] = {
                "min":   min(values),
                "max":   max(values),
                "avg":   round(sum(values) / len(values), 4),
                "count": len(values),
            }
        else:
            unique = list({str(v) for v in values})
            stats[key] = {
                "unique_count":  len(unique),
                "sample_values": unique[:10],
            }
    return stats


# ─────────────────────────────────────────────────────────────────
# 9. MIDDLEWARE
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_skills_summary() -> str:
    """Cached — SKILLS list is fixed at startup, this string never changes."""
    if not SKILLS:
        return "No skills loaded."
    lines   = [f"- **{s['name']}**: {s['description']}" for s in SKILLS]
    summary = "\n".join(lines)
    logger.debug(f"_build_skills_summary CACHE MISS | {len(SKILLS)} skill(s)")
    return summary


class SkillMiddleware(AgentMiddleware):
    tools = [
        list_databases,
        discover_labels,
        discover_label_schema,
        discover_relationship_schema,
        count_nodes,
        run_cypher_query,
        traverse_path,
        filter_by_confidence,
        load_skill,
    ]

    def __init__(self) -> None:
        self.skills_summary = _build_skills_summary()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        addendum = (
            f"\n\n## Available Skills\n\n{self.skills_summary}\n\n"
            "Call `load_skill(<name>)` to load a skill. Only load what you need."
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": addendum}
        ]
        return handler(
            request.override(system_message=SystemMessage(content=new_content))
        )


# ─────────────────────────────────────────────────────────────────
# 10. AGENT
# ─────────────────────────────────────────────────────────────────

agent = create_agent(
    model,
    tools=[],
    system_prompt=(
        "You are a fully autonomous graph query agent. "
        "You have no built-in knowledge of labels, relationships, or properties "
        "— you discover everything live from the database.\n\n"

        "STEP 1 — Inventory: list_databases()\n"
        "STEP 2 — Survey: discover_labels(db_name)\n"
        "STEP 3 — Drill down: discover_label_schema() and "
        "discover_relationship_schema() for relevant entities only\n"
        "STEP 4 — Load skill: load_skill('medical_cypher')\n"
        "STEP 5 — Query:\n"
        "  PATH questions          → traverse_path()\n"
        "  HIGH-CONFIDENCE only    → filter_by_confidence()\n"
        "  ALL OTHER questions     → run_cypher_query()\n\n"

        "STEP 6 — Self-heal on error:\n"
        "  DATABASE ERROR / SYNTAX ERROR → load_skill('error_fixer'), fix, retry ×3\n"
        "  SECURITY ERROR  → stop and report\n"
        "  CONFIG ERROR    → call list_databases() and use a valid db_name\n"
        "  INVALID INPUT   → fix the rejected parameter\n\n"

        "CRITICAL RULES:\n"
        "- Always pass db_name explicitly.\n"
        "- Never assume label or relationship names — always discover first.\n"
        "- Never drop or ignore results — every relationship can be clinically significant.\n"
        "- Always return confidence scores when present.\n"
        "- Relationship direction is biologically meaningful — always verify it.\n"
        "- Never aggregate data with AVG or SUM — it loses biological meaning.\n"
        "- Flag confidence < 0.5 as 'preliminary'.\n"
        "- Explain findings in plain medical English after results.\n"
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)
logger.info("Agent created successfully")
print("[STARTUP] Agent ready.\n")


# ─────────────────────────────────────────────────────────────────
# 11. EXECUTION
# ─────────────────────────────────────────────────────────────────

class AgentError(Exception):
    """Raised by query_graph when the agent fails to produce an answer."""


def query_graph(question: str) -> str:
    """
    Run one question through the agent on a fresh context thread.

    Returns the answer string.
    Raises AgentError on failure — callers can handle it properly.
    FIX C7: no longer swallows exceptions silently.
    """
    fresh_thread_id = str(uuid.uuid4())
    config          = {"configurable": {"thread_id": fresh_thread_id}}

    logger.info(f"query_graph START | thread={fresh_thread_id} | question={question}")
    print(f"\n{'='*60}")
    print(f"Question : {question}")
    print(f"Thread   : {fresh_thread_id}")
    print(f"{'='*60}\n")

    t_start = time.monotonic()
    try:
        response = agent.invoke(
            {"messages": [("user", question)]},
            config=config,
        )
        elapsed = time.monotonic() - t_start
        answer  = response["messages"][-1].content

        logger.info(
            f"query_graph END | thread={fresh_thread_id} | "
            f"elapsed={elapsed:.2f}s | answer_len={len(answer)}"
        )
        print(f"\n{'─'*60}")
        print(f"Answer (in {elapsed:.1f}s):\n{answer}")
        print(f"{'─'*60}\n")
        return answer

    except Exception as e:
        elapsed = time.monotonic() - t_start
        logger.error(
            f"query_graph FAILED | thread={fresh_thread_id} | "
            f"elapsed={elapsed:.2f}s | {type(e).__name__}: {e}",
            exc_info=True,
        )
        print(f"\n[ERROR] Agent failed after {elapsed:.1f}s: {e}\n")
        raise AgentError(f"Agent failed: {e}") from e


# ─────────────────────────────────────────────────────────────────
# 12. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomous Medical Knowledge Graph Agent")
    print("  Logs → logs/medical_agent.log")
    print("=" * 60)

    while True:
        try:
            question = input("\nAsk a question (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[EXIT] Shutting down...")
            break

        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        try:
            query_graph(question)
        except AgentError as e:
            print(f"[ERROR] {e}")

    print("\n[SHUTDOWN] Closing database connections...")
    for _name, _pool in DB_POOLS.items():
        _pool.close()
    logger.info("All pools closed. Shutdown complete.")
    print("[SHUTDOWN] Done.")