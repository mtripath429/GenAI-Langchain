# streamlit_app.py
import os
import sys
from typing import TypedDict, List, Literal, Dict, Any

import streamlit as st

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Tools (stable, non-experimental)
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.utilities import PythonREPL


# -------------------- Streamlit setup --------------------
st.set_page_config(page_title="LangGraph + Search", layout="centered")
st.title("LangGraph + Search (Wikipedia & DuckDuckGo)")

with st.sidebar:
    st.caption(f"Python: `{sys.executable}`")

# Read API key
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if not os.getenv("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY in Streamlit secrets or environment.")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  


# -------------------- Graph state --------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]


# -------------------- Build tools (stable paths) --------------------
def build_tools() -> List[Any]:
    tools: List[Any] = []

    # Wikipedia
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=3,
            doc_content_chars_max=2000,
        )
    )
    tools.append(wiki)

    # DuckDuckGo web search (no API key needed)
    tools.append(DuckDuckGoSearchRun(name="duckduckgo_search"))

    # Python REPL (wrap utility as a Tool)
    py = PythonREPL()
    python_tool = Tool.from_function(
        name="python_repl",
        description="Execute Python code and return printed output. Use print(...) to emit results.",
        func=py.run,
    )
    tools.append(python_tool)

    return tools


# -------------------- Build model --------------------
def build_model(tools: List[Any]) -> ChatOpenAI:
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return model.bind_tools(tools)


# -------------------- Graph nodes --------------------
def call_model(state: GraphState, config: RunnableConfig = None):
    model = config["configurable"]["model"]
    response = model.invoke(state["messages"], config=config)
    return {"messages": state["messages"] + [response]}


def call_tools(state: GraphState, config: RunnableConfig = None):
    tools_by_name: Dict[str, Any] = config["configurable"]["tools_by_name"]
    messages = state["messages"]
    last = messages[-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {"messages": messages}

    tool_messages: List[ToolMessage] = []
    for tc in last.tool_calls:
        name = tc["name"]
        args = tc.get("args") or {}
        tool = tools_by_name.get(name)
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Requested tool '{name}' not available.",
                    name=name,
                    tool_call_id=tc["id"],
                )
            )
            continue
        try:
            result = tool.invoke(args)
        except Exception as e:
            result = f"Tool '{name}' error: {repr(e)}"
        tool_messages.append(
            ToolMessage(
                content=str(result),
                name=name,
                tool_call_id=tc["id"],
            )
        )
    return {"messages": messages + tool_messages}


def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "__end__"


# -------------------- Compile graph --------------------
def compile_graph():
    tools = build_tools()
    model = build_model(tools)
    tools_by_name = {t.name: t for t in tools}

    graph = StateGraph(GraphState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "model")

    app = graph.compile(config_schema=None)
    cfg = RunnableConfig(configurable={"model": model, "tools_by_name": tools_by_name})
    return app, cfg


APP, CFG = compile_graph()


def run_chat(user_input: str):
    history = st.session_state.messages + [HumanMessage(content=user_input)]
    out = APP.invoke({"messages": history}, config=CFG)
    st.session_state.messages = out["messages"]


# -------------------- Simple chat UI --------------------
# Show history
for m in st.session_state.messages:
    role = "assistant" if isinstance(m, AIMessage) else ("user" if isinstance(m, HumanMessage) else "tool")
    with st.chat_message(role):
        # Keep tool outputs concise
        if isinstance(m, ToolMessage):
            st.markdown(f"**{m.name} →**")
            st.code(m.content[:1200])
        else:
            st.write(m.content)

# Input
prompt = st.chat_input("Ask: Who is the quarterback for the Bears?")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    run_chat(prompt)

    # Render the newest assistant message
    for m in reversed(st.session_state.messages):
        if isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.write(m.content)
            break
