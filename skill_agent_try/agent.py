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

# ─────────────────────────────────────────────
# 1. Initialize Database and Tools
# ─────────────────────────────────────────────
db = SQLDatabase.from_uri("sqlite:///business.db")
db.run("CREATE TABLE IF NOT EXISTS orders (id INT, total REAL, customer_id INT)")
db.run("INSERT OR IGNORE INTO orders VALUES (1, 150.50, 101)")

model = ChatOpenAI(model="gpt-4o", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()

# ─────────────────────────────────────────────
# 2. Load skill files from disk (metadata only at startup)
# ─────────────────────────────────────────────
def load_skills_from_directory(directory: str = "./skills") -> List[Dict]:
    """
    Reads all .md skill files and parses frontmatter + body.
    Returns a list of dicts with keys: name, description, content.
    """
    skills = []
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), "r") as f:
                raw = f.read()
            # Expect frontmatter between --- markers
            parts = raw.split("---", 2)
            if len(parts) == 3:
                meta = yaml.safe_load(parts[1])
                body = parts[2].strip()
            else:
                # No frontmatter — use filename as name
                meta = {"name": filename[:-3], "description": "No description."}
                body = raw.strip()

            skills.append({
                "name": meta.get("name", filename[:-3]),
                "description": meta.get("description", "No description."),
                "content": body,
            })
    return skills

SKILLS: List[Dict] = load_skills_from_directory("./skills")

# ─────────────────────────────────────────────
# 3. Dynamic load_skill TOOL  ← key change
#    The agent calls this tool on-demand.
#    Only the chosen skill's full content enters context.
# ─────────────────────────────────────────────
@tool
def load_skill(skill_name: str) -> str:
    """
    Load the full instructions of a skill into the agent's context.
    Call this BEFORE attempting a task that requires specialised knowledge.

    Args:
        skill_name: The exact name of the skill to load
                    (e.g. 'sql_generator', 'error_fixer').
    """
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"[Skill loaded: {skill_name}]\n\n{skill['content']}"

    available = ", ".join(s["name"] for s in SKILLS)
    return (
        f"Skill '{skill_name}' not found. "
        f"Available skills: {available}"
    )

# ─────────────────────────────────────────────
# 4. Middleware — injects DESCRIPTIONS only (not full content)
#    The agent sees what skills exist but NOT their full text.
#    Full text arrives only after the agent calls load_skill().
# ─────────────────────────────────────────────
class SkillMiddleware(AgentMiddleware):
    """
    Appends a lightweight skill directory to the system prompt.
    Registers load_skill as an available tool.
    """

    # Makes load_skill available to the agent
    tools = [load_skill]

    def __init__(self):
        skill_lines = [
            f"- **{s['name']}**: {s['description']}"
            for s in SKILLS
        ]
        self.skills_summary = "\n".join(skill_lines)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skill descriptions (not full content) into system prompt."""
        addendum = (
            f"\n\n## Available Skills\n\n"
            f"{self.skills_summary}\n\n"
            "Before working on a task, call `load_skill(<name>)` to load "
            "the relevant skill's full instructions into your context. "
            "Only load skills you actually need."
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": addendum}
        ]
        modified_request = request.override(
            system_message=SystemMessage(content=new_content)
        )
        return handler(modified_request)

# ─────────────────────────────────────────────
# 5. Create the agent
# ─────────────────────────────────────────────
agent = create_agent(
    model,
    tools=sql_tools,          # SQL execution tools
    system_prompt=(
        "You are a SQL query assistant. "
        "Use the available skills to understand schemas and fix errors. "
        "Always call load_skill() for a skill before relying on its instructions."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)

# ─────────────────────────────────────────────
# 6. Execution
# ─────────────────────────────────────────────
def query_database(question: str):
    config = {"configurable": {"thread_id": "user_session_123"}}
    response = agent.invoke(
        {"messages": [("user", question)]},
        config=config,
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    question = input("Enter your SQL query question: ")
    query_database(question)
