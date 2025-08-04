# state_act_agent.py
#
# Example implementation of the StateAct pattern for agentic workflows.
#
# The agent alternates between two phases:
#   1. State: The agent analyzes its current state, updates its plan and knowledge, and decides the next action.
#   2. Act: The agent executes the chosen action (tool), observes the result, and updates its state accordingly.
#
# This loop continues until the agent determines that its goal is achieved or no further actions are needed.
#
# In this example, the agent uses a plan-driven approach to solve a multi-step problem, leveraging tools (web search, calculator),
# and updating its state after each action. The agent's reasoning, plan, and knowledge are explicitly tracked and updated at each step.
#
# ---- ## ---- 

from typing import TypedDict, List, Optional, Dict

from pydantic import BaseModel, Field
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import tool

# --- Settings ---
OLLAMA_MODEL = "llama3.1:latest"

# --- State Definition ---
class AgentState(TypedDict):
    """The central state of the agent. Tracks the goal, plan, knowledge, tool outputs, and next action."""
    overall_goal: str
    plan: List[str]
    knowledge_summary: str
    tool_outputs: list
    question: str
    final_answer: Optional[str]
    action: Optional[str]
    action_input: Optional[Dict]

# --- Tool Definitions ---
@tool
# Example tool: Simulates a web search for factual information.
def web_search(query: str) -> str:
    """Use this to search for factual information on the web."""
    print(f"--- Executing Web Search for: {query} ---")
    q = query.lower()
    if "ponyvidia" in q and "market cap" in q:
        return "PNVDA's market cap is $3.1 Trillion ($3.1e12)."
    elif "vinland" in q and "gdp" in q:
        return "Vinland's GDP is $2.2 Trillion ($2.2e12)."
    else:
        return "No information found. Try a different, more specific query."

@tool
# Example tool: Executes a mathematical expression.
def python_calculator(expression: str) -> str:
    """Executes a pure Python mathematical expression."""
    print(f"--- Executing Python Calculator with: {expression} ---")
    try:
        safe_expression = "".join(c for c in expression if c in "0123456789.e+-*/() ")
        result = eval(safe_expression, {"__builtins__": None}, {})
        return f"Calculation result: {result:,.2f}"
    except Exception as e:
        return f"Error: {e}. Ensure the expression is a valid mathematical string."

tools = [web_search, python_calculator]

# --- LLM and Prompt Engineering ---
# Defines the expected output structure for the agent's reasoning and next action.
class AgentResponse(BaseModel):
    """The required JSON output structure for the agent's response."""
    thought: str = Field(description="Your reasoning and analysis of the current state.")
    plan: List[str] = Field(description="The updated plan. Mark steps with '[x]' only after success.")
    knowledge_summary: str = Field(description="Concise summary of all gathered facts and calculation results.")
    action: str = Field(description="The name of the next tool to use from the 'Toolbox' or 'finish'.")
    action_input: dict = Field(description="The dictionary input for the chosen action. E.g., {'query': '...'} or {'expression': '...'}")
    final_answer: Optional[str] = Field(description="The final answer to the user, only when the plan is fully complete.")

# LLM setup: Uses Ollama with a JSON output format for structured reasoning.
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json")
parser = PydanticOutputParser(pydantic_object=AgentResponse)

# Prompt template: Guides the LLM to follow the StateAct pattern and output structured reasoning.
prompt_template = PromptTemplate(
    template="""You are a methodical AI assistant. You must follow the plan and use the tools provided to answer the user's question.

**Toolbox:**
{tools}

**Current State:**
- Goal: {overall_goal}
- Plan: {plan}
- Knowledge Summary: {knowledge_summary}
- Last Tool Output: {tool_outputs}

**Instructions:**
1.  **Analyze**: Review the 'Last Tool Output' and your current state.
2.  **Update State**: Update your `knowledge_summary` and `plan` based on new information.
3.  **Decide Next Step**: Choose the next `action` and formulate the `action_input`.
4.  **Output**: You MUST provide your response in the following JSON format.

{format_instructions}
""",
    input_variables=["overall_goal", "plan", "knowledge_summary", "tool_outputs"],
    partial_variables={
        "tools": "\n".join([f"- {t.name}: {t.description}" for t in tools]),
        "format_instructions": parser.get_format_instructions()
    }
)

agent_runnable = prompt_template | llm | parser

# --- LangGraph Node Definitions ---
# Node: Runs the LLM to determine the next action and update the agent's state.
def run_agent(state: AgentState):
    print("\n--- 🧠 Running Agent Node ---")
    if all(step.startswith('[x]') for step in state['plan']) and state.get('final_answer'):
        return {
            **state,
            "action": "finish"
        }
    response = agent_runnable.invoke(state)
    print(f"LLM Response: {response}")
    return response.model_dump()

# Node: Executes the chosen tool and updates the tool outputs in the state.
def execute_tool(state: AgentState):
    """Executes the chosen tool."""
    print("\n--- 🛠️ Executing Tool Node ---")
    action = state.get("action")
    action_input = state.get("action_input")
    tool_map = {t.name: t for t in tools}
    if action not in tool_map:
        error_msg = f"Error: Tool '{action}' not found. Available tools are: {list(tool_map.keys())}"
        print(f"--- {error_msg} ---")
        return {"tool_outputs": [error_msg]}
    selected_tool = tool_map[action]
    try:
        output = selected_tool.invoke(action_input)
    except Exception as e:
        output = f"Error executing tool {action}: {e}"
        print(f"--- {output} ---")
    return {"tool_outputs": [output]}

# --- Graph Definition and Execution ---
# The StateGraph alternates between reasoning (run_agent) and acting (execute_tool),
# updating the state at each step, until the agent decides to finish.
graph = StateGraph(AgentState)
graph.add_node("run_agent", run_agent)
graph.add_node("execute_tool", execute_tool)
graph.set_entry_point("run_agent")

def route_action(state: AgentState):
    print("\n--- 🔀 Routing Action ---")
    action = state.get("action")
    if not action or action == 'finish':
        print("Action: finish. Ending graph.")
        return END
    else:
        print(f"Action: {action}. Executing tool.")
        return "execute_tool"

graph.add_conditional_edges("run_agent", route_action)
graph.add_edge("execute_tool", "run_agent")
app = graph.compile()

# --- Run the Agent ---
# Example: The agent is given a multi-step question and a plan to follow.
# It will alternate between reasoning and acting, updating its state and plan at each step.
initial_state = {
    "question": "Compare PonyVidia's market cap to the GDP of Vinland. What is the difference?",
    "overall_goal": "Compare PonyVidia's market cap to the GDP of Vinland and find the difference.",
    "plan": [
        "Find PonyVidia's current market cap.",
        "Find Vinland's recent GDP.",
        "Calculate the difference using a clean mathematical expression.",
        "Provide the final answer."
    ],
    "knowledge_summary": "No information gathered yet.",
    "tool_outputs": [],
}

print("--- 🚀 Starting Agent Execution ---")
final_state = {}
for s in app.stream(initial_state, {"recursion_limit": 15}):
    node_name = list(s.keys())[0]
    final_state.update(list(s.values())[0])
    print(s)

print("\n\n--- ✅ AGENT EXECUTION COMPLETE ---")
print(f"Final Answer: {final_state.get('final_answer')}")
print(f"Final Knowledge Summary: {final_state.get('knowledge_summary')}")