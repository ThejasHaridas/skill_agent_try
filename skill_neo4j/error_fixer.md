---
name: error_fixer
description: Diagnoses and fixes Cypher query errors and agent tool errors step by step
---

# Cypher Error Fixer

## Step-by-Step Diagnosis

### Step 1 — Identify the error type

The agent returns one of these error prefixes. Each has a different fix:

| Error Prefix | Meaning | Fix |
|---|---|---|
| `DATABASE ERROR` | Neo4j rejected the query | See Cypher error table below |
| `SYNTAX ERROR` | Bad Cypher syntax | Re-read query carefully, check table below |
| `SECURITY ERROR` | Query contains a blocked pattern | **Stop — do not rewrite. Report to user.** |
| `CONFIG ERROR` | Unknown db_name | Call `list_databases()`, use a valid name |
| `INVALID INPUT` | Bad label/property/filter name | Fix the rejected parameter (alphanumeric + underscore only) |
| `INVALID LABEL` | Label failed validation | Use only alphanumeric + underscore characters |
| `INVALID TYPE` | Relationship type failed validation | Use only alphanumeric + underscore characters |

---

### Step 2 — For DATABASE ERROR / SYNTAX ERROR: read the message carefully

| Message contains | Cause | Fix |
|---|---|---|
| `Variable not defined` | Used a variable before MATCH | Add MATCH clause for that variable |
| `Type mismatch` | Wrong type e.g. string vs int | Cast with `toString()` or `toInteger()` |
| `Property does not exist` | Wrong property name | Re-check with `discover_label_schema()` |
| `Label not found` | Wrong node label | Re-check with `discover_labels()` |
| `Relationship type not found` | Wrong rel type | Re-check with `discover_labels()` |
| `Syntax error` | Typo in Cypher | Read query character by character |
| `Authentication error` | Wrong credentials | Check environment variables. Cannot be fixed by rewriting. |
| `Retries exhausted` | Database unavailable | Wait and retry. Cannot be fixed by rewriting. |

---

### Step 3 — Re-check the schema

If a label or property name was wrong, call the correct tool again:
- Wrong label name → `discover_labels(db_name)`
- Wrong property name → `discover_label_schema(label, db_name)`
- Wrong relationship type or direction → `discover_relationship_schema(rel_type, db_name)`

**Neo4j is case-sensitive**: `Gene` ≠ `gene`, `TREATS` ≠ `treats`, `name` ≠ `Name`.

---

### Step 4 — Rewrite and retry

Fix only the specific error. Do not rewrite the entire query unless necessary.
Run the corrected query with `run_cypher_query()`.
Retry up to **3 times total**. After 3 failed attempts, stop and report:
- The last error message
- The last query attempted
- Which schema discovery calls were made

---

## Common Cypher Fixes

### Wrong property name
```cypher
-- After discover_label_schema() shows the property is 'identifier', not 'id'
-- ❌ Wrong
MATCH (n:Gene {id: 'BRCA1'}) RETURN n

-- ✅ Fixed
MATCH (n:Gene {identifier: 'BRCA1'}) RETURN n
```

### Wrong relationship direction
```cypher
-- After discover_relationship_schema() shows direction is Drug → Disease
-- ❌ Returns nothing (direction reversed)
MATCH (dis:Disease)-[:TREATS]->(d:Drug) RETURN d, dis

-- ✅ Fixed
MATCH (d:Drug)-[:TREATS]->(dis:Disease) RETURN d, dis
```

### Query returns too many rows — add specificity
```cypher
-- ❌ Too broad — may be slow or hit the large-result warning
MATCH (a)-[r]->(b) RETURN a, r, b

-- ✅ Add a WHERE clause to narrow results — do NOT just add LIMIT,
--   as limiting rows can silently drop clinically relevant data
MATCH (a:Drug)-[r]->(b:Disease)
WHERE a.name = 'Metformin'
RETURN a, r, b
```

> **Important**: Do NOT add a bare `LIMIT` to fix performance problems.
> In medical graphs, every result can be clinically significant.
> Narrow with WHERE clauses instead so no relevant data is silently discarded.