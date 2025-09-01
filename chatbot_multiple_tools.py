from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langchain_core.messages import AnyMessage

# Load environment variables
load_dotenv()

# Initialize tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv research papers.")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

tavily = TavilySearch()

# combine all these tools in the list
tools=[arxiv, wiki, tavily]

# Initialize LLM
llm = ChatGroq(model="qwen/qwen3-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# State schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


### Node definition
def tool_calling_llm(state:State) -> State:
    try:
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"Error occured while calling LLM: {str(e)}")
        return {"messages": [error_msg]}

# Build graph
builder = StateGraph(State)
# Nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

#Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",
# If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
                       tools_condition)
builder.add_edge("tools","tool_calling_llm")

graph = builder.compile()

# Invoke graph
query = "research paper 1706.03762 and What is the recent AI news and then please tell me the recent research paper on quantum computing?"
initial_state = {"messages": [HumanMessage(content=query)] }
response = graph.invoke(initial_state)

for message in response["messages"]:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    else:
        print(message.pretty_print())

