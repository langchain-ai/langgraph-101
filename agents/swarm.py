from agents.invoice_agent import invoice_tools, invoice_subagent_prompt
from agents.music_agent import music_tools
from agents.utils import llm

from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm


# Create our handoff tools between agents
transfer_to_invoice_agent_handoff_tool = create_handoff_tool(
    agent_name = "invoice_information_agent_with_handoff",
    description = "Transfer user to the invoice information agent that can help with invoice information"
)

transfer_to_music_catalog_agent_handoff_tool = create_handoff_tool(
    agent_name = "music_catalog_agent_with_handoff",
    description = "Transfer user to the music catalog information agent that can help with music catalog information"
)

# Add the handoff tools to the tools list
invoice_tools_with_handoff = [transfer_to_music_catalog_agent_handoff_tool] + invoice_tools
music_tools_with_handoff = [transfer_to_invoice_agent_handoff_tool] + music_tools

# Create the invoice information agent with handoff
invoice_information_agent_with_handoff = create_react_agent(
    llm,
    invoice_tools_with_handoff,
    prompt = invoice_subagent_prompt,
    name = "invoice_information_agent_with_handoff"
)

#
music_assistant_prompt ="""
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
    
    Message history is also attached.  
    """

music_catalog_agent_with_handoff = create_react_agent(
    llm,
    music_tools_with_handoff,
    prompt = music_assistant_prompt,
    name = "music_catalog_agent_with_handoff"
)


swarm_workflow = create_swarm(
    agents = [invoice_information_agent_with_handoff, music_catalog_agent_with_handoff],
    default_active_agent = "invoice_information_agent_with_handoff",
)

graph = swarm_workflow.compile(name="music_store_swarm")
