from agents.invoice_agent import graph as invoice_agent
from agents.music_agent import graph as music_agent
from agents.utils import llm

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: int
    loaded_memory: str
    remaining_steps: int

supervisor_prompt = """You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related question regarding past purchases, song or album availabilities. 
You are dedicated to providing exceptional service and ensuring customer queries are answered thoroughly, and have a team of subagents that you can use to help answer queries from customers. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers. Always respond to the customer through summarizing the conversation, including individual responses from subagents. 
If a question is unrelated to music or invoice, politely remind the customer regarding your scope of work. Do not answer unrelated answers. 

Your team is composed of two subagents that you can use to help answer the customer's request:
1. music_catalog_information_subagent: this subagent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
2. invoice_information_subagent: this subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. 

Based on the existing steps that have been taken in the messages, your role is to call the appropriate subagent based on the users query."""


@tool(
    name_or_callable="invoice_information_subagent",
    description="""
        An agent that can assistant with all invoice-related queries. It can retrieve information about a customers past purchases or invoices.
        """
)
def call_invoice_information_subagent(runtime: ToolRuntime, query: str):
    print('made it here')
    print(f"invoice subagent input: {query}")
    result = invoice_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "customer_id": runtime.state.get("customer_id", {})
    })
    subagent_response = result["messages"][-1].content
    return subagent_response

@tool(
    name_or_callable="music_catalog_subagent",
    description="""
        An agent that can assistant with all music-related queries. This agent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
        catalog (albums, tracks, songs, etc.) from the database. 
        """
)
def call_music_catalog_subagent(runtime: ToolRuntime, query: str):
    result = music_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "customer_id": runtime.state.get("customer_id", {})
    })
    subagent_response = result["messages"][-1].content
    return subagent_response

supervisor = create_agent(
    model="anthropic:claude-3-7-sonnet-latest", 
    tools=[call_invoice_information_subagent, call_music_catalog_subagent], 
    name="supervisor",
    system_prompt=supervisor_prompt, 
    state_schema=State, 
)
