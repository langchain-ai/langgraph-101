"""Deep Agent for LangGraph Studio.

A research agent built with DeepAgents that demonstrates:
- Custom tools (Tavily search + strategic thinking)
- Research subagent for delegated work
- Long-term memory via CompositeBackend (/memories/ -> StoreBackend)
- Human-in-the-loop on file writes

When running via `langgraph dev`, the store and checkpointer are
automatically provisioned by the platform.
"""

from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_core.tools import tool
from tavily import TavilyClient

from utils.models import model

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

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
"""

research_subagent = {
    "name": "research-agent",
    "description": "Delegate research tasks. Give one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS,
    "tools": [tavily_search, think_tool],
}


# --- Backend ---

def backend_factory(rt):
    """Route /memories/ to persistent StoreBackend, everything else ephemeral."""
    return CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": StoreBackend(rt),
        },
    )


# --- Agent ---

agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=f"""You are an expert research assistant. Today's date is {current_date}.

## Workflow
1. Use write_todos to plan your research
2. Delegate research to the research-agent using the task() tool
3. Synthesize findings into a comprehensive report
4. Write the final report to `/final_report.md`
5. Save key takeaways to `/memories/research_notes.md` for future reference

## Rules
- Delegate research to the research-agent rather than searching directly
- After receiving research results, synthesize and write the report yourself
- Consolidate citations (each unique URL gets one number)
- End reports with a Sources section

When referencing file paths, use backtick formatting like `path/file.md` instead of markdown links.
""",
    subagents=[research_subagent],
    backend=backend_factory,
    interrupt_on={
        "write_file": True,
        "edit_file": True,
    },
)
