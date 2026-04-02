"""MCP server exposing email and calendar tools via stdio transport.

Run directly: python -u mcp/email_tools.py
Used by notebooks as a subprocess via langchain-mcp-adapters.
"""

from mcp.server import FastMCP

mcp = FastMCP("Email Tools")


@mcp.tool(description="Write and send an email.")
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}'"


@mcp.tool(description="Check calendar availability for a given day.")
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


@mcp.tool(description="Schedule a calendar meeting.")
def schedule_meeting(attendees: str, subject: str, day: str, time: str) -> str:
    """Schedule a meeting."""
    return f"Meeting '{subject}' scheduled on {day} at {time} with {attendees}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
