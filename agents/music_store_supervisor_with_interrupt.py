from agents.invoice_agent import graph as invoice_agent
from agents.music_agent import graph as music_agent
from agents.music_store_supervisor import supervisor
from agents.utils import llm, get_engine_for_chinook_db

from langgraph.graph import StateGraph, START, END
from typing import Annotated, Optional, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
import ast
from langchain_community.utilities.sql_database import SQLDatabase

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class State(InputState):
    customer_id: NotRequired[str]
    loaded_memory: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]

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
            intent_message = AIMessage(
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
    return {"messages": [HumanMessage(content=user_input)]}


# conditional_edge
def should_interrupt(state: State, config: RunnableConfig):
    if state.get("customer_id") is not None:
        return "continue"
    else:
        return "interrupt"



# Add nodes 
multi_agent_verify = StateGraph(State, input_schema = InputState)
multi_agent_verify.add_node("verify_info", verify_info)
multi_agent_verify.add_node("human_input", human_input)
multi_agent_verify.add_node("supervisor", supervisor)

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
