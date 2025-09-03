from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from agents.utils import llm, get_engine_for_chinook_db
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
class State(InputState):
    customer_id: str
    loaded_memory: str
    remaining_steps: RemainingSteps 


# Create the SQL toolkit - this gives us all the tools we need to interact with the database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Get all the tools from the toolkit
music_tools = toolkit.get_tools()


# Get individual tools from our toolkit
get_schema_tool = next(tool for tool in music_tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in music_tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

list_tables_tool = next(tool for tool in music_tools if tool.name == "sql_db_list_tables")
check_query_tool = next(tool for tool in music_tools if tool.name == "sql_db_query_checker")


# Node 1: ALWAYS list tables first (no choice given to the model)
def list_tables(state: State):
    """This node automatically lists all available tables."""
    # Create a predetermined tool call - we're forcing this to happen
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "list_tables_call",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    
    # Execute the tool
    tool_message = list_tables_tool.invoke(tool_call)
    
    # Create a helpful response message
    response = AIMessage(f"I found these tables in the database: {tool_message.content}")
    
    return {"messages": [tool_call_message, tool_message, response]}


# Node 2: Force the model to get schemas for relevant tables
def call_get_schema(state: State):
    """This node forces the model to call the schema tool for relevant tables."""
    # Extract the user's question from the conversation
    user_question = state["messages"][0].content if state["messages"] else ""
    
    # Create a prompt asking which tables are relevant
    prompt = f"""Based on this question: '{user_question}'
    and these available tables from the database,
    decide which table schemas you need to see to answer the question.
    Call the sql_db_schema tool with the relevant table names."""
    
    # Force the model to use the schema tool (tool_choice="any" means it MUST use a tool)
    llm_with_schema = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_schema.invoke(state["messages"] + [HumanMessage(content=prompt)])
    
    return {"messages": [response]}


# Node 3: Generate the SQL query
def generate_query(state: State):
    """Generate a SQL query based on the schemas and question."""
    generate_query_prompt = f"""
You are an agent designed to interact with a SQL database.
Given the table schemas you've seen and the user's question, create a syntactically correct SQLite query.
    
Important rules:
- Limit results to at most 5 unless specified otherwise
- Only select relevant columns, not all columns
- Order by relevant columns to get interesting results
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
- If getting song recommendations return the song, artist, and album name for each song.

Example of how to write SQLite queries:
SELECT t.Name as Song, ar.Name as Artist, al.Title as Album
FROM Track t
JOIN Genre g ON t.GenreId = g.GenreId
JOIN Album al ON t.AlbumId = al.AlbumId
JOIN Artist ar ON al.ArtistId = ar.ArtistId
WHERE g.Name = 'Rock'
ORDER BY ar.Name, al.Title
LIMIT 5

Return the response in a nice format for the user to read.

Additional context is provided below: 

Prior saved user preferences: {state.get("loaded_memory", "None")}
    
Message history is also attached.  
    """
    
    system_message = SystemMessage(content=generate_query_prompt)
    
    # Bind the query tool but DON'T force its use - allow natural response if query is complete
    llm_with_query = llm.bind_tools([run_query_tool])
    response = llm_with_query.invoke([system_message] + state["messages"])
    
    return {"messages": [response]}


# Node 4: Check the query for common mistakes
def check_query(state: State):
    """Double-check the SQL query for common mistakes before executing."""
    check_query_prompt = """
    You are a SQL expert. Double check this SQLite query for common mistakes:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should be used  
    - Using BETWEEN for exclusive ranges
    - Data type mismatches
    - Proper column names for joins
    - Correct function arguments
    
    If there are mistakes, rewrite the query. Otherwise, reproduce the original query.
    You will call sql_db_query to execute the query after this check.
    """
    
    # Get the query from the last tool call
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        query = last_message.tool_calls[0]["args"].get("query", "")
        
        # Create a message asking to check the query
        check_message = HumanMessage(content=f"Check this query: {query}")
        
        # Force the model to call the run_query tool after checking
        llm_with_query = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_query.invoke([SystemMessage(content=check_query_prompt), check_message])
        
        # Preserve the original message ID to maintain conversation flow
        response.id = last_message.id
        
        return {"messages": [response]}
    
    # If no tool call found, just pass through
    return {"messages": []}


# Define the routing logic for after query generation
def route_after_query_generation(state: State) -> Literal["check_query", "end"]:
    """Decide whether to check the query or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls (a query was generated), check it
    if last_message.tool_calls:
        return "check_query"
    else:
        # No tool calls means the model has provided a final answer
        return "end"


# Build the enhanced graph
enhanced_sql_workflow = StateGraph(State)

# Add all our nodes
enhanced_sql_workflow.add_node("list_tables", list_tables)
enhanced_sql_workflow.add_node("call_get_schema", call_get_schema)
enhanced_sql_workflow.add_node("get_schema", get_schema_node)
enhanced_sql_workflow.add_node("generate_query", generate_query)
enhanced_sql_workflow.add_node("check_query", check_query)
enhanced_sql_workflow.add_node("run_query", run_query_node)

# Define the flow - this is where we enforce the workflow!
# Step 1: ALWAYS start by listing tables
enhanced_sql_workflow.add_edge(START, "list_tables")

# Step 2: After listing tables, get relevant schemas
enhanced_sql_workflow.add_edge("list_tables", "call_get_schema")

# Step 3: Execute the schema tool call
enhanced_sql_workflow.add_edge("call_get_schema", "get_schema")

# Step 4: Generate a query based on schemas
enhanced_sql_workflow.add_edge("get_schema", "generate_query")

# Step 5: Conditionally route - either check the query or finish
enhanced_sql_workflow.add_conditional_edges(
    "generate_query",
    route_after_query_generation,
    {
        "check_query": "check_query",  # If query generated, check it
        "end": END,                     # If final answer provided, end
    }
)

# Step 6: After checking, run the query
enhanced_sql_workflow.add_edge("check_query", "run_query")

# Step 7: After running, go back to generate (to create response or retry)
enhanced_sql_workflow.add_edge("run_query", "generate_query")

# Compile the enhanced agent
graph = enhanced_sql_workflow.compile(name="enhanced_music_catalog_agent")