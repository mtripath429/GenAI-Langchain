import os
import streamlit as st
from typing import TypedDict, List, Literal, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# ---------------- Config ----------------
st.set_page_config(page_title="Search-Enabled AI Assistant")
st.title("Search-Enabled AI Assistant")

# Ensure API keys are present
# Set these in your environment or .streamlit/secrets.toml
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

if not os.getenv("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY in env or Streamlit secrets.")
    st.stop()
if not os.getenv("TAVILY_API_KEY"):
    st.warning("TAVILY_API_KEY not set. Tavily search calls will fail.")

# ---------------- Session ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # type: List[BaseMessage]

# ---------------- State ----------------
class State(TypedDict):
    messages: List[BaseMessage]

# ---------------- Tools ----------------
search_tool = TavilySearchResults(max_results=3)

@tool
def search(query: str) -> str:
    """Search the internet for information (Tavily)."""
    # TavilySearchResults.invoke accepts a string or dict depending on version.
    # Here we pass the query string directly.
    results = search_tool.invoke(query)
    lines = []
    for r in results:
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        if url:
            lines.append(f"- {title}: {content}\n  {url}")
        else:
            lines.append(f"- {title}: {content}")
    return "\n".join(lines) if lines else "No results."

# ---------------- Model ----------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ---------------- Nodes ----------------
def router_node(state: State) -> State:
    """No-op node used as graph entrypoint; returns state unchanged."""
    return {"messages": state["messages"]}

def should_search(state: State) -> Literal["search", "respond"]:
    """Decide whether to search based on the last human message."""
    last_message = state["messages"][-1].content if state["messages"] else ""
    decision = model.invoke(
        [
            HumanMessage(
                content=(
                    "You are a routing assistant. "
                    "Given the user question below, decide if external search is needed. "
                    "Return exactly one word: 'search' or 'respond'.\n\n"
                    f"Question: {last_message}"
                )
            )
        ]
    )
    text = (decision.content or "").strip().lower()
    return "search" if "search" in text else "respond"

def search_step(state: State) -> State:
    """Run search and append a summarized AI message with the findings."""
    last_message = state["messages"][-1].content if state["messages"] else ""
    findings = search.invoke(last_message)  # call our @tool
    new_msg = AIMessage(content=f"I searched and found:\n\n{findings}")
    return {"messages": [*state["messages"], new_msg]}

def response_step(state: State) -> State:
    """Generate the final response using the conversation so far."""
    reply = model.invoke(state["messages"])
    return {"messages": [*state["messages"], reply]}

# ---------------- Graph ----------------
workflow = StateGraph(State)

workflow.add_node("router", router_node)      # entry node
workflow.add_node("search", search_step)
workflow.add_node("respond", response_step)

# Route based on LLM decision
workflow.add_conditional_edges(
    "router",
    should_search,
    {"search": "search", "respond": "respond"},
)

# If we searched, then respond
workflow.add_edge("search", "respond")

# Set entry point and compile
workflow.set_entry_point("router")
chain = workflow.compile()

# ---------------- UI ----------------
st.write("Ask me anything. I can search the web when needed.")

user_input = st.text_input("Your question:")

if user_input:
    # Run the graph with the new user message
    result = chain.invoke({"messages": [HumanMessage(content=user_input)]})

    # Render all messages from this run
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Assistant:** {msg.content}")
