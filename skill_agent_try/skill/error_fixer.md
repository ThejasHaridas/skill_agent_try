---
name: error_fixer
description: Handles execution, error analysis, and retracing steps.
---
# Evaluation & Error Fixing
1. **Execution**: Run the generated SQL using the `sql_db_query` tool.
2. **Error Handling**: If an error occurs, analyze the message. Retrace by checking the schema again. 
3. **Recovery**: Reformulate the query based on the error logic.
4. **Empty Results**: If the query returns nothing, check filters. If still empty, output "No data available."