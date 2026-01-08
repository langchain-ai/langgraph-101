import ast
from typing import Annotated, List, NotRequired, Optional

from aipe.llm import init_payx_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents.music_store.music_store_supervisor import supervisor
from utils.utils import get_engine_for_chinook_db

model = init_payx_chat_model(model="gpt-41", model_provider="azure_openai")

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)


class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class State(InputState):
    customer_id: NotRequired[str]
    loaded_memory: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]


class UserInput(BaseModel):
    """Schema for parsing user-provided account information."""

    identifier: str = Field(
        description="Identifier, which can be a customer ID, email, or phone number."
    )


structured_llm = model.with_structured_output(schema=UserInput)
structured_system_prompt = """You are a customer service representative responsible for extracting customer identifier.\n 
Only extract the customer's account information from the message history. 
If they haven't provided the information yet, return an empty string for the file"""


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


def verify_info(state: State):
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
        parsed_info = structured_llm.invoke(
            [SystemMessage(content=structured_system_prompt)] + [user_input]
        )

        # Extract details
        identifier = parsed_info.identifier

        customer_id = ""
        # Attempt to find the customer ID
        if identifier:
            customer_id = get_customer_id_from_identifier(identifier)

        if customer_id != "":
            intent_message = AIMessage(
                content=f"Thank you for providing your information! I was able to verify your account with customer id {customer_id}."
            )
            return {"customer_id": customer_id, "messages": [intent_message]}
        else:
            response = model.invoke(
                [SystemMessage(content=system_instructions)] + state["messages"]
            )
            return {"messages": [response]}

    else:
        pass


# Node
def human_input(state: State):
    """No-op node that should be interrupted on"""
    user_input = interrupt("Please provide input.")
    return {"messages": [HumanMessage(content=user_input)]}


# conditional_edge
def should_interrupt(state: State):
    if state.get("customer_id") is not None:
        return "continue"
    else:
        return "interrupt"


# helper function to structure memory
def format_user_memory(user_data):
    """Formats music preferences from users, if available."""
    profile = user_data["memory"]
    result = ""

    # Handle both Pydantic model (attributes) and dict (keys) representations
    if isinstance(profile, dict):
        music_prefs = profile.get("music_preferences", [])
    else:
        music_prefs = getattr(profile, "music_preferences", [])

    if music_prefs:
        result += f"Music Preferences: {', '.join(music_prefs)}"
    return result.strip()


# Node
def load_memory(state: State, store: BaseStore):
    """Loads music preferences from users, if available."""

    user_id = str(state["customer_id"])  # Convert to string to match create_memory
    namespace = ("memory_profile", user_id)
    existing_memory = store.get(namespace, "user_memory")
    formatted_memory = ""
    if existing_memory and existing_memory.value:
        formatted_memory = format_user_memory(existing_memory.value)

    return {"loaded_memory": formatted_memory}


# User profile structure for creating memory


class UserProfile(BaseModel):
    customer_id: str = Field(description="The customer ID of the customer")
    music_preferences: List[str] = Field(
        description="The music preferences of the customer"
    )


create_memory_prompt = """You are an expert analyst that is observing a conversation that has taken place between a customer and a customer support assistant. The customer support assistant works for a digital music store, and has utilized a multi-agent team to answer the customer's request. 
You are tasked with analyzing the conversation that has taken place between the customer and the customer support assistant, and updating the memory profile associated with the customer. 
You specifically care about saving any music interest the customer has shared about themselves, particularly their music preferences to their memory profile.

<core_instructions>
1. The memory profile may be empty. If it's empty, you should ALWAYS create a new memory profile for the customer.
2. You should identify any music interest the customer during the conversation and add it to the memory profile **IF** it is not already present.
3. For each key in the memory profile, if there is no new information, do NOT update the value - keep the existing value unchanged.
4. ONLY update the values in the memory profile if there is new information.
</core_instructions>

<expected_format>
The customer's memory profile should have the following fields:
- customer_id: the customer ID of the customer
- music_preferences: the music preferences of the customer

IMPORTANT: ENSURE your response is an object with these fields.
</expected_format>


<important_context>
**IMPORTANT CONTEXT BELOW**
To help you with this task, I have attached the conversation that has taken place between the customer and the customer support assistant below, as well as the existing memory profile associated with the customer that you should either update or create. 

The conversation between the customer and the customer support assistant that you should analyze is as follows:
{conversation}

The existing memory profile associated with the customer that you should either update or create based on the conversation is as follows:
{memory_profile}

</important_context>

Reminder: Take a deep breath and think carefully before responding.
"""


# Node
def create_memory(state: State, store: BaseStore):
    user_id = str(state["customer_id"])
    namespace = ("memory_profile", user_id)
    formatted_memory = state["loaded_memory"]
    formatted_system_message = SystemMessage(
        content=create_memory_prompt.format(
            conversation=state["messages"], memory_profile=formatted_memory
        )
    )
    # Anthropic requires at least one user message along with the system message
    user_prompt = HumanMessage(
        content="Please analyze the conversation and update the customer's memory profile according to the instructions."
    )
    updated_memory = model.with_structured_output(UserProfile).invoke(
        [formatted_system_message, user_prompt]
    )
    key = "user_memory"
    # Convert Pydantic model to dict to avoid pickle serialization issues on restart
    store.put(namespace, key, {"memory": updated_memory.model_dump()})


multi_agent_final = StateGraph(State, input_schema=InputState)
multi_agent_final.add_node("verify_info", verify_info)
multi_agent_final.add_node("human_input", human_input)
multi_agent_final.add_node("load_memory", load_memory)
multi_agent_final.add_node("supervisor", supervisor)
multi_agent_final.add_node("create_memory", create_memory)

multi_agent_final.add_edge(START, "verify_info")
multi_agent_final.add_conditional_edges(
    "verify_info",
    should_interrupt,
    {
        "continue": "load_memory",
        "interrupt": "human_input",
    },
)
multi_agent_final.add_edge("human_input", "verify_info")
multi_agent_final.add_edge("load_memory", "supervisor")
multi_agent_final.add_edge("supervisor", "create_memory")
multi_agent_final.add_edge("create_memory", END)

agent = multi_agent_final.compile(name="multi_agent_verify")
