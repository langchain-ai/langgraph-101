from agents.invoice_agent import graph as invoice_agent
from agents.music_agent import graph as music_agent
from agents.utils import llm

from langgraph.graph import StateGraph, START, END
from typing import Annotated, Optional, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
import ast
from langchain_community.utilities.sql_database import SQLDatabase
from agents.utils import get_engine_for_chinook_db

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

supervisor_prompt = """You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related question regarding past purchases, song or album availabilities. 
You are dedicated to providing exceptional service and ensuring customer queries are answered thoroughly, and have a team of subagents that you can use to help answer queries from customers. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers. Always respond to the customer through summarizing the conversation, including individual responses from subagents. 
If a question is unrelated to music or invoice, politely remind the customer regarding your scope of work. Do not answer unrelated answers. 

Your team is composed of two subagents that you can use to help answer the customer's request:
1. music_catalog_information_subagent: this subagent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
3. invoice_information_subagent: this subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. 

Based on the existing steps that have been taken in the messages, your role is to generate the next subagent that needs to be called. 
This could be one step in an inquiry that needs multiple sub-agent calls. """


from langgraph_supervisor import create_supervisor

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: NotRequired[str]
    loaded_memory: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]

# Create supervisor workflow
supervisor_prebuilt_workflow = create_supervisor(
    agents=[invoice_agent, music_agent],
    output_mode="last_message", # alternative is full_history
    model=llm,
    prompt=(supervisor_prompt), 
    state_schema=State
)

from pydantic import BaseModel, Field

class UserInput(BaseModel):
    """Schema for parsing user-provided account information."""
    identifier: str = Field(description = "Identifier, which can be a customer ID, email, or phone number.")


structured_llm = llm.with_structured_output(schema=UserInput)
structured_system_prompt = """You are a customer service representative responsible for extracting customer identifier.\n 
Only extract the customer's account information from the message history. 
If they haven't provided the information yet, return an empty string for the file"""

from typing import Optional

# Helper 
def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """
    Retrieve Customer ID using an identifier, which can be a customer ID, email, or phone number.
    
    Args:
        identifier (str): The identifier can be customer ID, email, or phone.
    
    Returns:
        Optional[int]: The CustomerId if found, otherwise None.
    """
    if identifier.isdigit():
        return int(identifier)
    elif identifier[0] == "+":
        query = f"SELECT CustomerId FROM Customer WHERE Phone = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    elif "@" in identifier:
        query = f"SELECT CustomerId FROM Customer WHERE Email = '{identifier}';"
        result = db.run(query)
        formatted_result = ast.literal_eval(result)
        if formatted_result:
            return formatted_result[0][0]
    return None 


# Node

def verify_info(state: State, config: RunnableConfig):
    """Verify the customer's account by parsing their input and matching it with the database."""

    if state.get("customer_id") is None: 
        system_instructions = """You are a music store agent, where you are trying to verify the customer identity 
        as the first step of the customer support process. 
        Only after their account is verified, you would be able to support them on resolving the issue. 
        In order to verify their identity, one of their customer ID, email, or phone number needs to be provided.
        If the customer has not provided their identifier, please ask them for it.
        If they have provided the identifier but cannot be found, please ask them to revise it."""

        user_input = state["messages"][-1] 
    
        # Parse for customer ID
        parsed_info = structured_llm.invoke([SystemMessage(content=structured_system_prompt)] + [user_input])
    
        # Extract details
        identifier = parsed_info.identifier
    
        customer_id = ""
        # Attempt to find the customer ID
        if (identifier):
            customer_id = get_customer_id_from_identifier(identifier)
    
        if customer_id != "":
            intent_message = SystemMessage(
                content= f"Thank you for providing your information! I was able to verify your account with customer id {customer_id}."
            )
            return {
                  "customer_id": customer_id,
                  "messages" : [intent_message]
                  }
        else:
          response = llm.invoke([SystemMessage(content=system_instructions)]+state['messages'])
          return {"messages": [response]}

    else: 
        pass

from langgraph.types import interrupt
# Node
def human_input(state: State, config: RunnableConfig):
    """ No-op node that should be interrupted on """
    user_input = interrupt("Please provide input.")
    return {"messages": [user_input]}


# conditional_edge
def should_interrupt(state: State, config: RunnableConfig):
    if state.get("customer_id") is not None:
        return "continue"
    else:
        return "interrupt"

supervisor_prebuilt = supervisor_prebuilt_workflow.compile(name="music_catalog_subagent")

# Add nodes 
multi_agent_verify = StateGraph(State, input_schema = InputState)
multi_agent_verify.add_node("verify_info", verify_info)
multi_agent_verify.add_node("human_input", human_input)
multi_agent_verify.add_node("supervisor", supervisor_prebuilt)

multi_agent_verify.add_edge(START, "verify_info")
multi_agent_verify.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {
        "continue": "supervisor",
        "interrupt": "human_input",
    },
)
multi_agent_verify.add_edge("human_input", "verify_info")
multi_agent_verify.add_edge("supervisor", END)
graph = multi_agent_verify.compile(name="multi_agent_verify")
