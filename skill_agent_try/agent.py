import os
import yaml
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver

# 1. Initialize Database and Tools
db = SQLDatabase.from_uri("sqlite:///business.db")
# Example data setup
db.run("CREATE TABLE IF NOT EXISTS orders (id INT, total REAL, customer_id INT)")
db.run("INSERT INTO orders VALUES (1, 150.50, 101)")

model = ChatOpenAI(model="gpt-4o", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 2. Define the SkillMiddleware
class SkillMiddleware:
    def __init__(self, directory: str = "skills"):
        self.directory = directory
        self.skills = self._load_skills()

    def _load_skills(self) -> List[Dict]:
        loaded_skills = []
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
        for filename in os.listdir(self.directory):
            if filename.endswith(".md"):
                with open(os.path.join(self.directory, filename), 'r') as f:
                    content = f.read()
                    # Splitting frontmatter from content
                    _, frontmatter, body = content.split('---', 2)
                    meta = yaml.safe_load(frontmatter)
                    loaded_skills.append({**meta, "content": body.strip()})
        return loaded_skills

    def __call__(self, state):
        """
        This hook modifies the state before the model is called.
        It injects the full content of relevant skills into the prompt.
        """
        skill_text = "\n\n".join([f"## Skill: {s['name']}\n{s['content']}" for s in self.skills])
        # Append the skill instructions to the current system prompt
        state["system_prompt"] += f"\n\nFOLLOW THESE SKILL RULES:\n{skill_text}"
        return state

# 3. Create the Agent
# The 'create_agent' function sets up the ReAct loop and applies middleware.
agent = create_agent(
    model,
    tools=tools,
    system_prompt=(
        "You are a SQL query assistant. You MUST use your skills to "
        "first understand the schema, then execute, and if an error occurs, "
        "retrace why it happened using the error_fixer instructions."
    ),
    middleware=[SkillMiddleware(directory="./skills")],
    checkpointer=InMemorySaver(), # Enables memory for iterative re-tracing
)

# 4. Execution Example
def query_database(question: str):
    # 'thread_id' is required for the checkpointer to maintain session memory
    config = {"configurable": {"thread_id": "user_session_123"}}
    
    # We invoke the agent. If the SQL fails, the agent will see the error,
    # refer to the 'error_fixer' skill in its prompt, and try again automatically.
    response = agent.invoke(
        {"messages": [("user", question)]}, 
        config=config
    )
    
    # Print the final result
    print(response["messages"][-1].content)

if __name__ == "__main__":
    question=input("Enter your SQL query question: ")
    query_database(question)