"""Deep Agent for LangGraph Studio.

A research agent built with DeepAgents that demonstrates:
- AGENTS.md for agent identity and instructions (replaces hardcoded system_prompt)
- Skills for on-demand capabilities (LinkedIn post, Twitter/X post)
- Custom tools (Tavily search + strategic thinking)
- Research subagent for delegated work
- Long-term memory via CompositeBackend (/memories/ -> StoreBackend)
- Human-in-the-loop on file writes

When running via `langgraph dev`, the store and checkpointer are
automatically provisioned by the platform.
"""

import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_core.tools import tool
from tavily import TavilyClient

from utils.models import model

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Tools ---

tavily_client = TavilyClient()


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: Search query to execute
    """
    search_results = tavily_client.search(query, max_results=3, topic="general")

    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = result.get("content", "No content available")
        result_text = f"## {title}\n**URL:** {url}\n\n{content}\n\n---\n"
        result_texts.append(result_text)

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n{''.join(result_texts)}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress.

    Use this after each search to analyze results and plan next steps.

    Args:
        reflection: Your detailed reflection on research progress
    """
    return f"Reflection recorded: {reflection}"


# --- Research Subagent ---

current_date = datetime.now().strftime("%Y-%m-%d")

RESEARCHER_INSTRUCTIONS = f"""You are a research assistant conducting research. Today's date is {current_date}.

<Task>
Use tools to gather information about the research topic.
</Task>

<Hard Limits>
- Simple queries: Use 2-3 search tool calls maximum
- Complex queries: Use up to 5 search tool calls maximum
- After each search, use think_tool to reflect on findings
</Hard Limits>

<Output Format>
Structure your findings with:
- Clear headings
- Inline citations [1], [2], [3]
- Sources section at the end
</Output Format>
"""

research_subagent = {
    "name": "research-agent",
    "description": "Delegate research tasks. Give one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS,
    "tools": [tavily_search, think_tool],
}


# --- Backend ---


def backend_factory(rt):
    """FilesystemBackend for disk access (skills, AGENTS.md), /memories/ routed to StoreBackend."""
    return CompositeBackend(
        default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
        routes={
            # Memories will be stored in the langgraph store, which is visible in studio by clicking the "memory" button.
            "/memories/": StoreBackend(rt),
        },
    )


# --- Agent ---

agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt="You are an expert research assistant.",
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=[research_subagent],
    backend=backend_factory,
    interrupt_on={
        "write_file": True,
        "edit_file": True,
    },
)

# approve an action in studio by entering: {"decisions": [{"type": "approve"}]} in the interrupt input.