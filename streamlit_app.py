# streamlit_app.py
import os
import sys
from typing import TypedDict, List, Literal, Dict, Any, Tuple

import streamlit as st

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool

from langchain_openai import ChatOpenAI

# Tools
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun  # optional web search
from langchain_community.utilities import PythonREPL  # we will wrap with Tool.from_function


# -------------------------- Streamlit page config --------------------------
st.set_page_config(
    page_title="LangGraph + Streamlit (Search + Python REPL)",
    page_icon=None,
    layout="centered",
)

st.title("LangGraph Chat with Search (Wikipedia / DuckDuckGo) + Python REPL")

# Show which Python executable Streamlit is actually using (helps when debugging envs)
with st.sidebar:
    st.caption(f"Python: `{sys.executable}`")


# -------------------------- Secrets / API keys --------------------------
# Prefer Streamlit secrets; fallback to env
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if not os.getenv("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY via Streamlit secrets or environment to run this app.")
    st.stop()


# -------------------------- Session state init --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # type: List[BaseMessage]

if "graph_bundle" not in st.session_state:
    st.session_state.graph_bundle = None  # {"app": app, "cfg": cfg}
if "graph_key" not in st.session_state:
    st.session_state.graph_key: Tuple[Any, ...] = tuple()


# -------------------------- Sidebar controls --------------------------
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Use a model that supports tool/function calling.",
    )
    temperature = st.slider("Temperature", 0.0, 1.2, 0.2, 0.1)

    st.markdown("**Tools**")
    use_wikipedia = st.checkbox("Wikipedia", value=True)
    use_ddg = st.checkbox("DuckDuckGo (web search)", value=True, help="No API key needed.")
    use_python = st.checkbox("Python REPL", value=True, help="Execute small Python snippets.")

    show_tool_traces = st.checkbox("Show tool traces", value=False)

    if st.button("Reset conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.experimental_rerun()


# -------------------------- LangGraph plumbing --------------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]


def build_tools() -> List[Any]:
    """Construct the tool list based on sidebar toggles."""
    tools: List[Any] = []

    if use_wikipedia:
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=3,
                doc_content_chars_max=2000,
            )
        )
        tools.append(wiki)

    if use_ddg:
        # Name set explicitly so the tool-call name is stable
        tools.append(DuckDuckGoSearchRun(name="duckduckgo_search"))

    if use_python:
        # Use PythonREPL utility and expose it as a Tool
        py = PythonREPL()
        python_tool = Tool.from_function(
            name="python_repl",
            description=(
                "Execute Python code in a REPL. Returns printed output. "
                "Use print(...) to emit results."
            ),
            func=py.run,
        )
        tools.append(python_tool)

    return tools


def build_model(tools: List[Any]) -> ChatOpenAI:
    """Create a tool-capable LLM and bind tools."""
    model = ChatOpenAI(model=model_name, temperature=temperature)
    return model.bind_tools(tools)


def call_model(state: GraphState, config: RunnableConfig = None):
    """Run the LLM and let it decide if tools are needed."""
    model = config["configurable"]["model"]
    response = model.invoke(state["messages"], config=config)
    return {"messages": state["messages"] + [response]}


def call_tools(state: GraphState, config: RunnableConfig = None):
    """Execute tool calls requested by the last AI message."""
    tools_by_name: Dict[str, Any] = config["configurable"]["tools_by_name"]
    messages = state["messages"]
    last = messages[-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
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
    """If the last AI message requested tools, run them; otherwise end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "__end__"


def compile_graph():
    """Compile a new graph based on current settings and return (app, cfg)."""
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


# Rebuild the graph when settings change (simple cache key)
current_key = (model_name, float(temperature), bool(use_wikipedia), bool(use_ddg), bool(use_python))
if st.session_state.graph_bundle is None or st.session_state.graph_key != current_key:
    st.session_state.graph_bundle = {}
    app, cfg = compile_graph()
    st.session_state.graph_bundle["app"] = app
    st.session_state.graph_bundle["cfg"] = cfg
    st.session_state.graph_key = current_key


def run_chat(user_input: str):
    """Invoke the graph with the current message history + new user input."""
    app = st.session_state.graph_bundle["app"]
    cfg = st.session_state.graph_bundle["cfg"]
    history = st.session_state.messages + [HumanMessage(content=user_input)]
    out = app.invoke({"messages": history}, config=cfg)
    st.session_state.messages = out["messages"]


# -------------------------- Chat UI --------------------------
# Render previous messages
for m in st.session_state.messages:
    role = (
        "assistant"
        if isinstance(m, AIMessage)
        else ("user" if isinstance(m, HumanMessage) else "tool")
    )
    with st.chat_message(role):
        if isinstance(m, ToolMessage):
            if show_tool_traces:
                st.markdown(f"**{m.name} →**")
                st.code(m.content)
        else:
            st.write(m.content)

# Input box
prompt = st.chat_input("Ask something like: Who is the quarterback for the Bears?")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    run_chat(prompt)

    # Render just the most recent assistant response (+ optional tool traces that came right before it)
    last_ai_shown = False
    for m in reversed(st.session_state.messages):
        if isinstance(m, AIMessage) and not last_ai_shown:
            with st.chat_message("assistant"):
                st.write(m.content)
            last_ai_shown = True
        elif isinstance(m, ToolMessage) and show_tool_traces and not last_ai_shown:
            with st.chat_message("tool"):
                st.markdown(f"**{m.name} →**")
                st.code(m.content)
        if last_ai_shown:
            break
