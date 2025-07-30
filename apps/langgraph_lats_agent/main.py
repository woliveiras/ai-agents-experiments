# langgraph_lats_agent.py
import gc
import json
import asyncio
from functools import partial
from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from ollama_utils import get_ollama_models

# --- 1. State Definition ---
class AgentState(TypedDict):
    messages: List[BaseMessage]

# --- 2. Tool Definition ---
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    print(f"--- ACTION: Searching for '{query}' ---")
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
    return json.dumps(results) if results else "[]"

# --- 3. Agent Nodes ---
async def call_agent_node(state: AgentState, model_with_tools: ChatOllama):
    """Nó do agente: pensa e decide a próxima ação."""
    print("\n--- AGENT: Thinking... ---")
    response = await model_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

async def call_tools_node(state: AgentState):
    """Nó de ferramentas: executa a ferramenta escolhida pelo agente."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "search_web":
        result = await search_web.ainvoke(tool_args)
        tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        return {"messages": [tool_message]}
    return {"messages": []}

# --- 4. Conditional Edge ---
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "end"
    return "continue"

# --- 5. Graph Construction ---
async def run_lats_agent(goal: str):
    # Use get_ollama_models to check available models
    available_models = get_ollama_models()
    print(f"Available Ollama models: {available_models}")
    model = ChatOllama(model="qwen3:latest")
    model_with_tools = model.bind_tools(tools=[search_web])

    # Graph definition
    workflow = StateGraph(AgentState)

    agent_node_with_model = partial(call_agent_node, model_with_tools=model_with_tools)

    workflow.add_node("agent", agent_node_with_model)
    workflow.add_node("tools", call_tools_node)

    # Define edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # --- Running the Agent ---
    print(f"--- GOAL: {goal} ---")
    initial_state = {"messages": [HumanMessage(content=goal)]}

    async for output in app.astream(initial_state):
        for key, value in output.items():
            print(f"--- Output from node '{key}' ---")
            print(value)
            print("-" * 30)

if __name__ == "__main__":
    user_goal = "What is the main concept behind the LATS framework for AI agents?"
    asyncio.run(run_lats_agent(user_goal))
    gc.collect()