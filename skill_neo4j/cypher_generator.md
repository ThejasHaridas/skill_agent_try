---
name: medical_cypher
description: Cypher query rules and patterns for medical knowledge graphs — schema is always discovered live, never assumed
---

# Medical Knowledge Graph — Cypher Rules

## Core Principle

Never assume node labels, relationship types, or property names.
Always discover them first:
1. `discover_labels(db_name)` — see all labels and relationship types
2. `discover_label_schema(label, db_name)` — see properties of a specific label
3. `discover_relationship_schema(rel_type, db_name)` — see direction and properties

The schema in this file is illustrative only. Use what the database actually has.

---

## Relationship Properties to Always Check

After discovering a relationship type with `discover_relationship_schema()`,
check if it carries evidence properties. Common ones across medical databases:

- `confidence`  (0.0 to 1.0 — most common)
- `score`       (varies by source)
- `evidence`    ('experimental', 'predicted', 'curated')
- `source`      ('DrugBank', 'STRING', 'DisGeNET', etc.)

Always include these in your `RETURN` clause — they are clinically significant.
Flag any result with `confidence < 0.5` as **preliminary**.

---

## Query Patterns

### Traverse a relationship you discovered

Replace `LabelA`, `LabelB`, `REL_TYPE`, `<property>`, and `<value>`
with what `discover_label_schema()` and `discover_relationship_schema()` returned.
Never use `.name` or any other property name by assumption.

```cypher
MATCH (a:LabelA {<property>: '<value>'})-[r:REL_TYPE]->(b:LabelB)
RETURN b.<property>, r.confidence, r.evidence
ORDER BY r.confidence DESC
```

### Neighbourhood of one entity — directed outgoing

Use `-->` or `<--` to get directed relationships only.
Never use `-[r]-` (undirected) — it mixes incoming and outgoing
relationships without distinguishing direction, which is clinically misleading.

```cypher
-- ✅ Outgoing only — biologically correct direction
MATCH (n:Label {<property>: '<value>'})-[r]->(neighbour)
RETURN type(r)          AS relationship,
       labels(neighbour) AS neighbour_type,
       neighbour.<property> AS neighbour_name,
       r.confidence     AS confidence
ORDER BY r.confidence DESC

-- ✅ Incoming only — if you need the reverse direction
MATCH (n:Label {<property>: '<value>'})<-[r]-(neighbour)
RETURN type(r)          AS relationship,
       labels(neighbour) AS neighbour_type,
       neighbour.<property> AS neighbour_name,
       r.confidence     AS confidence
```

### Filter by confidence after discovery

```cypher
MATCH (a:LabelA)-[r:REL_TYPE]->(b:LabelB)
WHERE r.confidence >= 0.7
RETURN a.<property>, b.<property>, r.confidence
```

Or use the `filter_by_confidence()` tool — it handles the injection automatically.

### Multi-hop path

Use `traverse_path()` tool for path questions — it returns structured chain
output with confidence at each hop, which is easier to interpret than raw path
objects. Use raw Cypher only when `traverse_path()` cannot express the query.

```cypher
MATCH path = (a:LabelA {<property>: '<value>'})-[*1..3]->(b:LabelB)
RETURN path, length(path) AS hops
ORDER BY hops ASC
LIMIT 20
```

---

## Rules

- **Never aggregate** with `AVG`, `SUM`, `COUNT` as a final answer — it loses
  biological meaning. Counting result sizes is acceptable; clinical conclusions are not.
- **Always return** `confidence`/`score` properties on relationships when they exist.
- **Relationship direction is biologically meaningful** — always verify with
  `discover_relationship_schema()` before writing a MATCH clause.
- **Never use undirected** `-[r]-` patterns — use `-->` or `<--` explicitly.
- **Never assume property names** — use only what `discover_label_schema()` returns.
  Do not default to `.name` without confirming it exists.
- **Use `traverse_path()`** for multi-hop questions — it handles path formatting
  and confidence reporting automatically.
- **Use `filter_by_confidence()`** when the question asks for well-evidenced
  or high-confidence associations only.