from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from agents.utils import llm, get_engine_for_chinook_db
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
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

# Bind tools to our LLM - this allows the model to call these tools when needed
llm_with_music_tools = llm.bind_tools(music_tools)

# Create a tool node that can execute our SQL tools
music_tool_node = ToolNode(music_tools)



# SQL assistant prompt
def generate_music_assistant_prompt(memory: str = "None") -> str:
    return f"""
You are a member of the music store assistant team, specifically focused on helping customers discover and learn about music in our digital catalog. You have access to a comprehensive music database containing information about Albums, Artists, Tracks, Genres, Playlists, and more.

CORE RESPONSIBILITIES:
- Search and provide accurate information about songs, albums, artists, and playlists
- Offer relevant music recommendations based on customer interests and preferences
- Handle music-related queries with attention to detail and expertise
- Help customers discover new music they might enjoy
- Generate syntactically correct SQLite queries to retrieve music catalog information

SEARCH GUIDELINES:
1. Always perform thorough searches before concluding something is unavailable
2. If exact matches aren't found, try:
   - Checking for alternative spellings or similar artist names
   - Looking for partial matches in song or album titles
   - Searching by genre or related artists
   - Checking different versions, remixes, or compilations
3. When providing music lists:
   - Include the artist name with each song/album
   - Mention the album when listing songs
   - Group results logically (by artist, genre, or album)
   - Limit results to 5 unless user specifies otherwise

SQL QUERY BEST PRACTICES:
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.
- Always start by examining available tables (Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track)
- Query relevant table schemas before writing complex queries
- Use JOINs to connect related information (e.g., Track → Album → Artist)
- Order results by relevance (popularity, alphabetical, or chronological)
- ALWAYS double-check queries before executing
- DO NOT make DML statements (INSERT, UPDATE, DELETE, DROP)
- Limit queries to 5 maximum to avoid making the user wait

MUSIC DATABASE STRUCTURE:
- Artists have Albums, Albums contain Tracks
- Tracks have Genres and can be in Playlists
- Use proper JOINs to get complete information (Track.Name, Album.Title, Artist.Name)

If you cannot find specific music in our catalog, politely inform the customer and suggest alternatives or similar artists that we do have available.

Additional context is provided below:

Prior saved user preferences: {memory}
    
Message history is also attached.
"""

# Node 
def music_assistant(state: State): 

    # Fetching long term memory
    memory = "None" 
    if "loaded_memory" in state: 
        memory = state["loaded_memory"]

    # Instructions for our agent  
    sql_assistant_prompt = generate_music_assistant_prompt(memory)

    # Invoke the model with the system prompt and conversation history
    response = llm_with_music_tools.invoke([SystemMessage(sql_assistant_prompt)] + state["messages"])
    
    # Update the state with the response
    return {"messages": [response]}


# Conditional edge that determines whether to continue or not
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Create the workflow for our SQL agent
music_workflow = StateGraph(State)

# Add nodes to our graph
music_workflow.add_node("music_assistant", music_assistant)
music_workflow.add_node("music_tool_node", music_tool_node)

# Add edges to define the flow
# First, we define the start node. The query will always route to the sql_assistant first
music_workflow.add_edge(START, "music_assistant")

# Add a conditional edge from sql_assistant
music_workflow.add_conditional_edges(
    "music_assistant",
    # Function representing our conditional edge
    should_continue,
    {
        # If there are tool calls, execute them
        "continue": "music_tool_node",
        # Otherwise we're done
        "end": END,
    },
)

# After executing tools, go back to the assistant to process results
music_workflow.add_edge("music_tool_node", "music_assistant")

# Compile the graph into an executable agent
graph = music_workflow.compile(name="music_assistant")