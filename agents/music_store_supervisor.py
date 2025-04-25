from agents.invoice_agent import graph as invoice_agent
from agents.music_agent import graph as music_agent
from agents.utils import llm

from langgraph.graph import StateGraph, START, END
from typing import Annotated, Optional
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict

supervisor_prompt = """You are an expert customer support assistant for a digital music store. 
You are dedicated to providing exceptional service and ensuring customer queries are answered thoroughly. 
You have a team of subagents that you can use to help answer queries from customers. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers. 

Your team is composed of two subagents that you can use to help answer the customer's request:
1. music_catalog_information_subagent: this subagent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
3. invoice_information_subagent: this subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. 

Based on the existing steps that have been taken in the messages, your role is to generate the next subagent that needs to be called. 
This could be one step in an inquiry that needs multiple sub-agent calls. """


from langgraph_supervisor import create_supervisor

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: Optional[str]
    loaded_memory: Optional[str]
    remaining_steps: Optional[RemainingSteps]

# Create supervisor workflow
supervisor_prebuilt_workflow = create_supervisor(
    agents=[invoice_agent, music_agent],
    output_mode="last_message", # alternative is full_history
    model=llm,
    prompt=(supervisor_prompt), 
    state_schema=State
)

graph = supervisor_prebuilt_workflow.compile(name="music_catalog_subagent")
