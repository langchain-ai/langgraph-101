from agents.utils import model, get_engine_for_chinook_db
from langchain_community.utilities.sql_database import SQLDatabase
from typing_extensions import TypedDict
from typing import Annotated, NotRequired
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import ToolNode
from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
import ast

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: NotRequired[str]

class State(InputState):
    customer_id: NotRequired[str]
    remaining_steps: NotRequired[RemainingSteps]



@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True
    )

@tool
def get_songs_by_genre(genre: str):
    """
    Fetch songs from the database that match a specific genre.
    
    Args:
        genre (str): The genre of the songs to fetch.
    
    Returns:
        list[dict]: A list of songs that match the specified genre.
    """
    genre_id_query = f"SELECT GenreId FROM Genre WHERE Name LIKE '%{genre}%'"
    genre_ids = db.run(genre_id_query)
    if not genre_ids:
        return f"No songs found for the genre: {genre}"
    genre_ids = ast.literal_eval(genre_ids)
    genre_id_list = ", ".join(str(gid[0]) for gid in genre_ids)

    songs_query = f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName
        FROM Track
        LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE Track.GenreId IN ({genre_id_list})
        GROUP BY Artist.Name
        LIMIT 8;
    """
    songs = db.run(songs_query, include_columns=True)
    if not songs:
        return f"No songs found for the genre: {genre}"
    formatted_songs = ast.literal_eval(songs)
    return [
        {"Song": song["SongName"], "Artist": song["ArtistName"]}
        for song in formatted_songs
    ]

@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_title}%';
        """,
        include_columns=True
    )

music_tools = [get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs]
llm_with_music_tools = model.bind_tools(music_tools)


# Node
music_tool_node = ToolNode(music_tools)

# Node 
def music_assistant(state: State): 

    # Fetching long term memory. 
    memory = "None" 
    if "loaded_memory" in state: 
        memory = state["loaded_memory"]

    # Intructions for our agent  
    music_assistant_prompt = f"""
    You are a member of the assistant team, your role specifically is to focused on helping customers discover and learn about music in our digital catalog. 
    If you are unable to find playlists, songs, or albums associated with an artist, it is okay. 
    Just inform the customer that the catalog does not have any playlists, songs, or albums associated with that artist.
    You also have context on any saved user preferences, helping you to tailor your response. 
    
    CORE RESPONSIBILITIES:
    - Search and provide accurate information about songs, albums, artists, and playlists
    - Offer relevant recommendations based on customer interests
    - Handle music-related queries with attention to detail
    - Help customers discover new music they might enjoy
    - You are routed only when there are questions related to music catalog; ignore other questions. 
    
    SEARCH GUIDELINES:
    1. Always perform thorough searches before concluding something is unavailable
    2. If exact matches aren't found, try:
       - Checking for alternative spellings
       - Looking for similar artist names
       - Searching by partial matches
       - Checking different versions/remixes
    3. When providing song lists:
       - Include the artist name with each song
       - Mention the album when relevant
       - Note if it's part of any playlists
       - Indicate if there are multiple versions
    
    Additional context is provided below: 

    Prior saved user preferences: {memory}
    
    Message history is also attached.  
    """

    # Invoke the model
    response = llm_with_music_tools.invoke([SystemMessage(music_assistant_prompt)] + state["messages"])
    
    # Update the state
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

music_workflow = StateGraph(State, input_schema = InputState)

# Add nodes 
music_workflow.add_node("music_assistant", music_assistant)
music_workflow.add_node("music_tool_node", music_tool_node)


# Add edges 
# First, we define the start node. The query will always route to the subagent node first. 
music_workflow.add_edge(START, "music_assistant")

# We now add a conditional edge
music_workflow.add_conditional_edges(
    "music_assistant",
    # Function representing our conditional edge
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "music_tool_node",
        # Otherwise we finish.
        "end": END,
    },
)



music_workflow.add_edge("music_tool_node", "music_assistant")

graph = music_workflow.compile(name="music_catalog_subagent")