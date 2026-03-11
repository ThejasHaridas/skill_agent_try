---
name: sql_generator
description: Handles DB schema loading and initial SQL draft creation.
---
# SQL Generation & Schema Loader
1. **Schema Discovery**: Call `sql_db_list_tables` first.
2. **Table Context**: Call `sql_db_schema` for the identified tables.
3. **Logic**: Write a SQL query based on the schema and the user's NLP question.