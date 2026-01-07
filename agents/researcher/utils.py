"""Simplified utility functions for the Deep Research agent."""

import logging
from datetime import datetime

from langchain_core.messages import (
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.tools import tool

from agents.researcher.models import ResearchComplete

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress.

    Use this tool after each search to analyze results and plan next steps.

    Args:
        reflection: Detailed reflection on research progress and next steps

    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"


##########################
# Tool Utils
##########################

async def get_all_tools():
    """Assemble complete toolkit for research operations.

    Returns tools including OpenAI's native web search.
    """
    # Core research tools
    tools = [tool(ResearchComplete), think_tool]

    # Add OpenAI's native web search
    # This is a special tool definition that OpenAI models recognize
    tools.append({"type": "web_search_preview"})

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used."""
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        return web_search_requests > 0

    except (AttributeError, TypeError):
        return False


def openai_websearch_called(response):
    """Detect if OpenAI's web search was used."""
    try:
        tool_outputs = response.additional_kwargs.get("tool_outputs")
        if not tool_outputs:
            return False

        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True

        return False
    except (AttributeError, TypeError):
        return False


async def execute_tool_safely(tool, args):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"
