# streamlit_app.py
import os
from typing import TypedDict, List, Literal, Dict, Any

import streamlit as st

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from langchain_openai import ChatOpenAI

# Tools
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun  # optional
from langchain_experimental.tools.python.tool import PythonREPLTool


# ---------- Streamlit page config ----------
st.set_page_config(page_title="LangGraph + Streamlit (Wikipedia/DuckDuckGo Search)", page_icon=None)
st.title("LangGraph Chat with Search (no Tavily)")

# ---------- Secrets / API keys ----------
# Prefer Streamlit secrets; fallback to environment variable
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if not os.getenv("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY in Streamlit secrets or environment to run this app.")
    st.stop()


# ---------- App state ----------
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "cfg" not in st.session_state:
    st.session_state.cfg = None


# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Must support tool/function calling."
    )
    temperature = st.slider("Temperature", 0.0, 1.2, 0.2, 0.1)

    st.markdown("**Tools**")
    use_wikipedia = st.checkbox("Wikipedia", value=True)
    use_ddg = st.checkbox("DuckDuckGo (web search)", value=True, help="No API key needed.")
    use_python = st.checkbox("Python REPL", value=True)

    show_tool_traces = st.checkbox("Show tool traces", value=False)

    if st.button("Reset conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.graph_ready = False
        st.session_state.cfg = None
        st.experimental_rerun()


# ---------- LangGraph plumbing ----------
class GraphState(TypedDict):
    messages: List[BaseMessage]


def build_tools() -> List[Any]:
    tools = []

    if use_wikipedia:
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=3,
                doc_content_chars_max=2000,
            )
        )
        tools.append(wiki)

    if use_ddg:
        tools.append(DuckDuckGoSearchRun(name="duckduckgo_search"))

    if use_python:
        tools.append(PythonREPLTool())

    return tools


def build_model(tools: List[Any]) -> ChatOpenAI:
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )
    return model.bind_tools(tools)


def call_model(state: GraphState, config: RunnableConfig = None):
    model = config["configurable"]["model"]
    response = model.invoke(state["messages"], config=config)
    return {"messages": state["messages"] + [response]}


def call_tools(state: GraphState, config: RunnableConfig = None):
    tools_by_name: Dict[str, Any] = config["configurable"]["tools_by_name"]
    messages = state["messages"]
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"messages": messages}

    tool_messages: List[ToolMessage] = []
    for tc in last.tool_calls:
        name = tc["name"]
        args = tc.get("args") or {}
        if name not in tools_by_name:
            tool_messages.append(
                ToolMessage(
                    content=f"Requested tool '{name}' not available.",
                    name=name,
                    tool_call_id=tc["id"],
                )
            )
            continue
        tool = tools_by_name[name]
        try:
            result = tool.invoke(args)
        except Exception as e:
            result = f"Tool '{name}' raised error: {repr(e)}"

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
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"


def ensure_graph():
    """Build and cache the compiled graph + config in session_state."""
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
    st.session_state.graph_ready = True
    st.session_state.cfg = {"app": app, "cfg": cfg}


# Build graph lazily (or after settings change)
if not st.session_state.graph_ready:
    ensure_graph()


def run_chat(user_input: str):
    """Invoke the graph with the current message history + new user input."""
    app = st.session_state.cfg["app"]
    cfg = st.session_state.cfg["cfg"]

    # Append the human message
    history = st.session_state.messages + [HumanMessage(content=user_input)]
    out = app.invoke({"messages": history}, config=cfg)
    st.session_state.messages = out["messages"]  # keep full transcript


# ---------- Chat UI ----------
# Render history
for m in st.session_state.messages:
    role = "assistant" if isinstance(m, AIMessage) else ("user" if isinstance(m, HumanMessage) else "tool")
    with st.chat_message(role):
        if isinstance(m, ToolMessage):
            if show_tool_traces:
                st.markdown(f"**{m.name} →**")
                st.code(m.content)
        else:
            st.write(m.content)

# Input box
prompt = st.chat_input("Ask me something like: Who is the quarterback for the Bears?")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    run_chat(prompt)
    # Render only the last AI message now
    for m in reversed(st.session_state.messages):
        if isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.write(m.content)
            break
        if isinstance(m, ToolMessage) and show_tool_traces:
            with st.chat_message("tool"):
                st.markdown(f"**{m.name} →**")
                st.code(m.content)
