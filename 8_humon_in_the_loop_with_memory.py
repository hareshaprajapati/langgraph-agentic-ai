from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import  StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage

load_dotenv()

@tool
def get_stock_price(symbol:str) -> float:
    """
    Return the current stock price of a given stock symbol.
    :param symbol: stock symbol
    :return: current stock price
    """
    price = {
        "MSFT": 200.3,
        "AAPL": 100.0,
        "GOOG": 200.0,
        "AMZN": 100.0,
    }.get(symbol.upper())
    if price is None:
        raise ValueError(f"Stock {symbol} not found.")
    return price

@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    """Buy stocks given the stock symbol and quantity."""
    decision = interrupt({
            "action": "buy_stocks",
            "symbol": symbol,
            "quantity": quantity,
            "total_price": total_price,
        })
    if str(decision).lower().startswith("y"):
        return f"You bought {quantity} shares of {symbol} for a total price of ${total_price:.2f}"
    else:
        return "You declined to buy."

tools = [get_stock_price, buy_stocks]
llm_with_tools = (init_chat_model("google_genai:gemini-2.5-flash")
                  .bind_tools(tools))

class State(TypedDict):
    # add_messages is a reducer that ensures messages are appended
    messages: Annotated[list[AnyMessage],add_messages]

def chatbot(state: State) -> State:
    try:
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

def logger(state: State) -> State:
    print("DEBUG:", state["messages"][-1].content)
    return state

builder = StateGraph(State)

builder.add_node("chatbot_node", chatbot)
builder.add_node("logger", logger)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START,"chatbot_node")
builder.add_conditional_edges("chatbot_node",
                              tools_condition,
                              path_map={ "tools" : "tools", "__end__" : "logger"}
                              )
builder.add_edge("tools","chatbot_node")
builder.add_edge("logger", END)

memory_saver = MemorySaver()

graph = builder.compile(checkpointer=memory_saver)

config = { "configurable": { "thread_id": "1" }}

message = HumanMessage(content="What is the current price of 10 MSFT stocks?")

response = graph.invoke({"messages": [message]}, config=config)
print(response["messages"][-1].content)

message = HumanMessage(content="Buy 10 MSFT stocks at current price.")

response = graph.invoke({"messages": [message]}, config=config)
print(response["messages"][-1].content)
print(response['__interrupt__'])
decision = input("Approve (yes/no): ")
response = graph.invoke(Command(resume=decision), config=config)
print(response["messages"][-1].content)

#%%
