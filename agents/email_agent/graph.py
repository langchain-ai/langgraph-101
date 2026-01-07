from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from dotenv import load_dotenv
from utils.models import model

load_dotenv("../.env")

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

llm_router = model.with_structured_output(RouterSchema) 

# Tools
@tool
def schedule_meeting(
    attendees: list[str], subject: str, duration_minutes: int, preferred_day: datetime, start_time: int
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return f"Meeting '{subject}' scheduled on {date_str} at {start_time} for {duration_minutes} minutes with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

@tool
class Done(BaseModel):
    """E-mail has been sent."""
    done: bool

tools = [schedule_meeting, check_calendar_availability, write_email, Done]
tools_by_name = {tool.name: tool for tool in tools}

llm_with_tools = model.bind_tools(tools, tool_choice="any", parallel_tool_calls=False)


# State definitions
class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict

class State(MessagesState):
    # This state class has the messages key build in
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]


# ------------------------------------------------------------
# Email Agent
# ------------------------------------------------------------
action_instructions = """
< Role >
You are a top-notch executive assistant who cares about helping your executive perform as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day, start_time) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. Done - E-mail has been sent

Note: FOR EACH INPUT, ONLY EVER CALL ONE TOOL
</ Tools >

< Instructions >
When handling emails, follow these steps:
1. Carefully analyze the email content and purpose
3. For responding to the email, draft a response email with the write_email tool
4. For meeting requests, use the check_calendar_availability tool to find open time slots
5. To schedule a meeting, use the schedule_meeting tool with a datetime object for the preferred_day parameter
   - Today's date is {today} - use this for scheduling meetings accurately
6. If you scheduled a meeting, then draft a short response email using the write_email tool
7. After using the write_email tool, the task is complete
8. If you have sent the email, then use the Done tool to indicate that the task is complete
</ Instructions >

< Background >
I'm Robert, a software engineer at LangChain.
</ Background >

< Response Preferences >
Use professional and concise language. If the e-mail mentions a deadline, make sure to explicitly acknowledge and reference the deadline in your response.

When responding to technical questions that require investigation:
- Clearly state whether you will investigate or who you will ask
- Provide an estimated timeline for when you'll have more information or complete the task

When responding to event or conference invitations:
- Always acknowledge any mentioned deadlines (particularly registration deadlines)
- If workshops or specific topics are mentioned, ask for more specific details about them
- If discounts (group or early bird) are mentioned, explicitly request information about them
- Don't commit 

When responding to collaboration or project-related requests:
- Acknowledge any existing work or materials mentioned (drafts, slides, documents, etc.)
- Explicitly mention reviewing these materials before or during the meeting
- When scheduling meetings, clearly state the specific day, date, and time proposed.

When responding to meeting scheduling requests:
- If the recipient is asking for a meeting commitment, verify availability for all time slots mentioned in the original email and then commit to one of the proposed times based on your availability by scheduling the meeting. Or, say you can't make it at the time proposed.
- If availability is asked for, then check your calendar for availability and send an email proposing multiple time options when available. Do NOT schedule meetings
- Mention the meeting duration in your response to confirm you've noted it correctly.
- Reference the meeting's purpose in your response.
</ Response Preferences >

< Calendar Preferences >
30 minute meetings are preferred, but 15 minute meetings are also acceptable.
Times later in the day are preferable.
</ Calendar Preferences >
"""

# Nodes
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""
    agent_system_prompt = action_instructions
    return {
        "messages": [
            llm_with_tools.invoke([
                {"role": "system", "content": agent_system_prompt.format(today=datetime.now().strftime("%Y-%m-%d"))}
            ] + state["messages"])
        ]
    }

def tool_node(state: State):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """Route to Action, or end if Done tool called"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "Action"

# Build workflow
agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("agent", llm_call)
agent_builder.add_node("tools", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "agent")
agent_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "tools",
        END: END,
    },
)
agent_builder.add_edge("tools", "agent")

# Compile the agent
agent = agent_builder.compile()

# ------------------------------------------------------------
# Triage Router
# ------------------------------------------------------------
def parse_email(email_input: dict) -> dict:
    """Parse an email input dictionary.

    Args:
        email_input (dict): Dictionary containing email fields:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content

    Returns:
        tuple[str, str, str, str]: Tuple containing:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content
    """
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )

def format_email_markdown(subject, author, to, email_thread, email_id=None):
    """Format email details into a nicely formatted markdown string for display
    
    Args:
        subject: Email subject
        author: Email sender
        to: Email recipient
        email_thread: Email content
        email_id: Optional email ID (for Gmail API)
    """
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    
    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}{id_section}

{email_thread}

---
"""

triage_instructions = """
< Role >
Your role is to triage incoming emails based upon instructs and background information below.
</ Role >

< Background >
I'm Robert, a software engineer at LangChain.
</ Background >

< Instructions >
Categorize each email into one of three categories:
1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that worth notification but doesn't require a response
3. RESPOND - Emails that need a direct response
Classify the below email into one of these categories.
</ Instructions >

< Rules >
Emails that are not worth responding to:
- Marketing newsletters and promotional emails
- Spam or suspicious emails
- CC'd on FYI threads with no direct questions

There are also other things that should be known about, but don't require an email response. For these, you should notify (using the `notify` response). Examples of this include:
- Team member out sick or on vacation
- Build system notifications or deployments
- Project status updates without action items
- Important company announcements
- FYI emails that contain relevant information for current projects
- HR Department deadline reminders
- GitHub notifications

Emails that are worth responding to:
- Direct questions from team members requiring expertise
- Meeting requests requiring confirmation
- Critical bug reports related to team's projects
- Requests from management requiring acknowledgment
- Client inquiries about project status or features
- Technical questions about documentation, code, or APIs (especially questions about missing endpoints or features)
- Personal reminders related to family (wife / daughter)
- Personal reminder related to self-care (doctor appointments, etc)
</ Rules >
"""

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """
    Analyze email content to decide if we should respond, notify, or ignore.
    """
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_instructions

    user_prompt = """
Please determine how to handle the below email thread:

From: {author}
To: {to}
Subject: {subject}
{email_thread}""".format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Run the router LLM
    result = model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    if classification == "respond":
        goto = "response_agent"
        # Add the email to the messages
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif result.classification == "ignore":
        update =  { "classification_decision": result.classification}
        goto = END
    elif result.classification == "notify":
        update = { "classification_decision": result.classification}
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

# Build workflow
overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
)
graph = overall_workflow.compile()