import os
import yaml
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable

# ─────────────────────────────────────────────────────────────────
# 1. CREDENTIALS — passed at runtime, nothing hardcoded
# ─────────────────────────────────────────────────────────────────
def get_db_uri() -> str:
    """
    Accepts DB credentials from environment variable or user input.
    Supports any SQLAlchemy URI:
      sqlite:///mydb.db
      postgresql://user:pass@host:5432/dbname
      mysql+pymysql://user:pass@host/dbname
    """
    uri = os.environ.get("DB_URI")
    if not uri:
        print("\nNo DB_URI environment variable found.")
        print("Examples:")
        print("  sqlite:///business.db")
        print("  postgresql://user:pass@localhost:5432/mydb")
        uri = input("Enter your database URI: ").strip()
    return uri

db = SQLDatabase.from_uri(get_db_uri())

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY") or input("Enter OpenAI API key: ")
)

toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()

# ─────────────────────────────────────────────────────────────────
# 2. SKILL LOADER — reads .md files from ./skills directory
# ─────────────────────────────────────────────────────────────────
def load_skills_from_directory(directory: str = "./skills") -> List[Dict]:
    skills = []
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), "r") as f:
                raw = f.read()

            parts = raw.split("---", 2)
            if len(parts) == 3:
                meta = yaml.safe_load(parts[1])
                body = parts[2].strip()
            else:
                meta = {"name": filename[:-3], "description": "No description."}
                body = raw.strip()

            skills.append({
                "name": meta.get("name", filename[:-3]),
                "description": meta.get("description", "No description."),
                "content": body,
            })
    return skills

SKILLS: List[Dict] = load_skills_from_directory("./skills")

# ─────────────────────────────────────────────────────────────────
# 3. AGENTIC TOOLS
#    The agent calls these itself — no human guidance needed
# ─────────────────────────────────────────────────────────────────

@tool
def discover_schema() -> str:
    """
    AUTO-DISCOVER the full database schema.

    Call this FIRST before writing any SQL query.
    Returns all table names, column names, types, and constraints
    directly from the live database — no prior knowledge needed.
    """
    tables = db.get_usable_table_names()
    if not tables:
        return "No tables found in the database."

    result = [f"Database contains {len(tables)} table(s): {', '.join(tables)}\n"]
    for table in tables:
        info = db.get_table_info(table_names=[table])
        result.append(f"### {table}\n{info}")

    return "\n\n".join(result)


@tool
def load_skill(skill_name: str) -> str:
    """
    Load the full instructions of a skill into context ON DEMAND.

    Call this when you need specialised guidance:
    - Call load_skill('sql_generator') before writing SQL
    - Call load_skill('error_fixer') when a query fails

    Only loads ONE skill at a time — keeps context lean.

    Args:
        skill_name: Exact skill name e.g. 'sql_generator', 'error_fixer'
    """
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"[Skill loaded: {skill_name}]\n\n{skill['content']}"

    available = ", ".join(s["name"] for s in SKILLS)
    return f"Skill '{skill_name}' not found. Available: {available}"


@tool
def check_query_safety(query: str) -> str:
    """
    Check if a SQL query is safe to execute (no DROP, DELETE, TRUNCATE etc).

    Always call this BEFORE executing any write operation.

    Args:
        query: The SQL query string to check
    """
    dangerous = ["DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT"]
    query_upper = query.upper()

    flagged = [kw for kw in dangerous if kw in query_upper]
    if flagged:
        return (
            f"UNSAFE QUERY DETECTED. "
            f"Contains: {', '.join(flagged)}. "
            f"This agent is read-only. Rewrite as a SELECT query."
        )
    return "Query is safe to execute."


# ─────────────────────────────────────────────────────────────────
# 4. MIDDLEWARE — injects skill descriptions only (not full content)
# ─────────────────────────────────────────────────────────────────
class SkillMiddleware(AgentMiddleware):
    """
    Appends lightweight skill descriptions to system prompt.
    Full skill content only enters context when agent calls load_skill().
    """

    # Registers these tools so the agent can call them
    tools = [discover_schema, load_skill, check_query_safety]

    def __init__(self):
        skill_lines = [
            f"- **{s['name']}**: {s['description']}"
            for s in SKILLS
        ]
        self.skills_summary = "\n".join(skill_lines) if skill_lines else "No skills loaded."

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        addendum = (
            f"\n\n## Available Skills\n\n"
            f"{self.skills_summary}\n\n"
            "Call `load_skill(<n>)` to load a skill's full instructions. "
            "Only load what you need."
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": addendum}
        ]
        modified_request = request.override(
            system_message=SystemMessage(content=new_content)
        )
        return handler(modified_request)


# ─────────────────────────────────────────────────────────────────
# 5. AGENT — autonomous loop, no human guidance needed
# ─────────────────────────────────────────────────────────────────
agent = create_agent(
    model,
    tools=sql_tools,
    system_prompt=(
        "You are a fully autonomous SQL agent. "
        "When given a question, follow this loop WITHOUT asking for help:\n\n"

        "STEP 1 — Discover: Call discover_schema() to understand all tables "
        "and columns in the live database.\n"

        "STEP 2 — Load skills: Call load_skill('sql_generator') to load "
        "query writing rules.\n"

        "STEP 3 — Safety check: Call check_query_safety(query) before "
        "executing any write operation.\n"

        "STEP 4 — Execute: Run the SQL query using the available SQL tools.\n"

        "STEP 5 — Self-heal: If the query fails, call load_skill('error_fixer'), "
        "diagnose the error, rewrite the query, and retry. "
        "Repeat up to 3 times before reporting failure.\n\n"

        "RULES:\n"
        "- Never ask the user for table names, column names, or schema info.\n"
        "- Never hardcode assumptions about the schema — always discover first.\n"
        "- Never run destructive SQL (DROP, DELETE, TRUNCATE).\n"
        "- Always explain your final answer in plain English after the result.\n"
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)


# ─────────────────────────────────────────────────────────────────
# 6. EXECUTION
# ─────────────────────────────────────────────────────────────────
def query_database(question: str, thread_id: str = "default_session"):
    """
    Run a natural language question against the database.
    The agent handles everything: schema discovery, SQL generation,
    error fixing, and result explanation.

    Args:
        question:  Natural language question e.g. "What are total sales last month?"
        thread_id: Session ID for conversation memory (default: 'default_session')
    """
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n Question: {question}\n")

    response = agent.invoke(
        {"messages": [("user", question)]},
        config=config,
    )

    print("-" * 60)
    print("Answer:")
    print(response["messages"][-1].content)
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomous SQL Agent")
    print("  Give it a question — it figures out the rest.")
    print("=" * 60)

    while True:
        question = input("\nAsk a question (or 'exit' to quit): ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if question:
            query_database(question)
