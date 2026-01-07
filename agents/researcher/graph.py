"""Simplified Deep Research agent implementation for educational purposes.

This is a streamlined version of the research agent with hardcoded configuration
for easier understanding and use in educational settings.
"""

import asyncio
import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.researcher.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from agents.researcher.models import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from agents.researcher.utils import (
    get_all_tools,
    get_notes_from_tool_calls,
    get_today_str,
    openai_websearch_called,
    think_tool,
)

from dotenv import load_dotenv

load_dotenv("../../.env")

# ===== HARDCODED CONFIGURATION =====
# These values are hardcoded for simplicity in this educational example
RESEARCH_MODEL = "openai:gpt-4.1"
MAX_RESEARCHER_ITERATIONS = 3  # Number of research supervisor iterations
MAX_REACT_TOOL_CALLS = 10  # Max tool calls per researcher
MAX_CONCURRENT_RESEARCH_UNITS = 5  # Max parallel research units
MAX_STRUCTURED_OUTPUT_RETRIES = 3  # Retry attempts for structured outputs
MAX_OUTPUT_TOKENS = 10000  # Max tokens for model outputs


def get_model():
    """Get or create the model instance."""
    return init_chat_model(
        model=RESEARCH_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


async def clarify_with_user(state: AgentState, config: RunnableConfig):
    """Ask clarifying questions if needed using human-in-the-loop."""

    messages = state["messages"]

    # Configure model for structured clarification analysis
    clarification_model = (
        get_model()
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # If clarification is needed, use interrupt to get user input
    if response.need_clarification:
        user_response = interrupt(response.question)
        return {"messages": [AIMessage(content=response.question), HumanMessage(content=user_response)]}
    else:
        # No clarification needed, add verification and continue
        return {"messages": [AIMessage(content=response.verification)]}


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor."""

    # Configure model for structured research question generation
    research_model = (
        get_model()
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers."""

    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # Configure model with tools and retry logic
    research_model = (
        get_model()
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    # Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor."""

    # Extract current state and check exit conditions
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > MAX_RESEARCHER_ITERATIONS
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # Limit concurrent research units
            allowed_conduct_research_calls = conduct_research_calls[:MAX_CONCURRENT_RESEARCH_UNITS]
            overflow_conduct_research_calls = conduct_research_calls[MAX_CONCURRENT_RESEARCH_UNITS:]

            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Exceeded max concurrent research units ({MAX_CONCURRENT_RESEARCH_UNITS})",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # Handle errors by ending research phase
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )

    # Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


# Supervisor Subgraph Construction
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics."""

    researcher_messages = state.get("researcher_messages", [])

    # Get all available research tools
    tools = await get_all_tools()
    if len(tools) == 0:
        raise ValueError("No tools found for research. Please configure search API.")

    # Prepare system prompt
    researcher_prompt = research_system_prompt.format(
        mcp_prompt="",  # Simplified - no MCP support
        date=get_today_str()
    )

    # Configure model with tools and retry logic
    research_model = (
        get_model()
        .bind_tools(tools)
        .with_retry(stop_after_attempt=MAX_STRUCTURED_OUTPUT_RETRIES)
    )

    # Generate researcher response
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool, args):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher."""

    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # Early exit if no tool calls
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # Execute all tool calls
    tools = await get_all_tools()
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"])
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # Check exit conditions
    exceeded_iterations = state.get("tool_call_iterations", 0) >= MAX_REACT_TOOL_CALLS
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # Continue research loop
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise summary."""

    researcher_messages = state.get("researcher_messages", [])

    # Add compression instruction
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # Create compression prompt
    compression_prompt = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages

    # Execute compression
    response = await get_model().ainvoke(messages)

    # Extract raw notes
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": str(response.content),
        "raw_notes": [raw_notes_content]
    }


# Researcher Subgraph Construction
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report."""

    # Extract research findings
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # Attempt report generation
    max_retries = 3
    current_retry = 0

    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            # Generate the final report
            final_report = await get_model().ainvoke([HumanMessage(content=final_report_prompt)])

            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            current_retry += 1
            continue

    # Return failure result
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed")],
        **cleared_state
    }


# Main Deep Researcher Graph Construction
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState
)

# Add main workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Define main workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "research_supervisor")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the complete deep researcher workflow
graph = deep_researcher_builder.compile()
