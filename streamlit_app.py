import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Configure page
st.set_page_config(page_title="Search-Enabled AI Assistant", page_icon="🔍")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define state schema
class State(TypedDict):
    messages: Sequence[BaseMessage]
    next: str

# Set up tools
search_tool = TavilySearchResults(max_results=3)

@tool
def search(query: str) -> str:
    """Search the internet for information."""
    results = search_tool.invoke(query)
    return "\n".join(f"- {result['title']}: {result['content']}" for result in results)

# Configure model
model = ChatOpenAI(temperature=0.7)

# Define nodes
def should_search(state: State) -> Union[str, Sequence[str]]:
    """Determine if we should search based on the last message."""
    last_message = state["messages"][-1].content
    
    response = model.invoke(
        [
            HumanMessage(content=f"""Based on this question: '{last_message}'
            Should I search for information to answer it? Reply with 'search' or 'no_search'.""")
        ]
    )
    
    return "search" if "search" in response.content.lower() else "respond"

def search_step(state: State) -> State:
    """Perform search and add results to messages."""
    last_message = state["messages"][-1].content
    search_results = search.invoke(last_message)
    
    new_message = AIMessage(content=f"I found these search results:\n{search_results}")
    return {"messages": [*state["messages"], new_message], "next": "respond"}

def response_step(state: State) -> State:
    """Generate final response."""
    response = model.invoke(state["messages"])
    return {"messages": [*state["messages"], response], "next": END}

# Create workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("should_search", should_search)
workflow.add_node("search", search_step)
workflow.add_node("respond", response_step)

# Add edges
workflow.add_edge("should_search", "search")
workflow.add_edge("should_search", "respond")
workflow.add_edge("search", "respond")

# Compile workflow
chain = workflow.compile()

# Streamlit interface
st.title("🔍 Search-Enabled AI Assistant")
st.write("Ask me anything! I can search the internet for up-to-date information.")

# Get user input
user_input = st.text_input("Your question:", key="user_input")

if user_input:
    # Run the chain
    result = chain.invoke({
        "messages": [HumanMessage(content=user_input)],
        "next": "should_search"
    })
    
    # Display all messages
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            st.write("You:", msg.content)
        elif isinstance(msg, AIMessage):
            st.write("Assistant:", msg.content)