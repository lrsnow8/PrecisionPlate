import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import NutritionState
from agent.prompts import get_system_prompt
from db.database import get_daily_summary
from tools.meal_logger import log_meal_text
from tools.meal_logger_vision import log_meal_photo
from tools.daily_summary import get_daily_summary as get_daily_summary_tool
from tools.goal_manager import set_goal, get_goal
from tools.nutrition_lookup import get_nutrition_info
from tools.meal_recommendations import get_meal_recommendation
from tools.historical_report import get_historical_report

# All tools registered with the chatbot
ALL_TOOLS = [
    log_meal_text,
    log_meal_photo,
    get_daily_summary_tool,
    set_goal,
    get_goal,
    get_nutrition_info,
    get_meal_recommendation,
    get_historical_report,
]

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.environ.get("CLAUDE_API_KEY"),
            max_tokens=1024,
        )
    return _llm


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 6   # max consecutive chatbot→tools cycles per user turn
MAX_TURN_MESSAGES = 8  # max new messages added this turn before forcing summarize


def load_context(state: NutritionState) -> dict:
    """Refresh today's macro snapshot from SQLite and inject into state."""
    user_id = state.get("user_id", "1")
    snapshot = get_daily_summary(user_id)
    # Snapshot the current message count so we can measure per-turn growth.
    # Reset the tool-call round counter for the new turn.
    return {
        "today_snapshot": snapshot,
        "tool_call_rounds": 0,
        "turn_start_msg_count": len(state.get("messages", [])),
    }


def chatbot(state: NutritionState) -> dict:
    """Call Claude with the system prompt + full message history."""
    summary = state.get("summary", "")
    today_snapshot = state.get("today_snapshot", {})
    system_prompt = get_system_prompt(summary, today_snapshot)

    llm_with_tools = _get_llm().bind_tools(ALL_TOOLS)

    # Build a LOCAL list for the LLM call — never written back to state.
    # This avoids "Please continue." being persisted as a real human message
    # and triggering runaway summarize → chatbot → summarize cycles.
    local_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    if not isinstance(local_messages[-1], HumanMessage):
        local_messages.append(HumanMessage(content="Please continue."))

    response = llm_with_tools.invoke(local_messages)
    return {"messages": [response]}


def summarize(state: NutritionState) -> dict:
    """
    Compress older messages into state.summary.
    Fires when len(messages) > 20; retains the last 10 raw messages.
    """
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # Messages to compress: everything except the last 10
    to_compress = messages[:-10]
    retained = messages[-10:]

    summary_prompt = (
        "You are summarizing a conversation between a user and a nutrition assistant. "
        "Produce a concise factual summary covering: goals set, meals logged, "
        "key nutrition insights discussed, and any outstanding user requests. "
        "Be brief — 3 to 8 sentences.\n\n"
    )
    if existing_summary:
        summary_prompt += f"Existing summary to extend:\n{existing_summary}\n\n"

    summary_prompt += "New messages to incorporate:\n"
    for msg in to_compress:
        role = getattr(msg, "type", "unknown")
        summary_prompt += f"{role}: {msg.content}\n"

    response = _get_llm().invoke([HumanMessage(content=summary_prompt)])
    new_summary = response.content

    return {
        "summary": new_summary,
        "messages": retained,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _turn_msg_delta(state: NutritionState) -> int:
    """Number of messages added since the start of the current user turn."""
    return len(state.get("messages", [])) - state.get("turn_start_msg_count", 0)


def should_continue(state: NutritionState):
    """Conditional edge from chatbot node."""
    last_message = state["messages"][-1]
    rounds = state.get("tool_call_rounds", 0)
    if last_message.tool_calls and rounds < MAX_TOOL_ROUNDS:
        return "tools"
    # Summarize only when this *turn* has grown the history significantly,
    # not based on the total accumulated message count.
    if _turn_msg_delta(state) > MAX_TURN_MESSAGES:
        return "summarize"
    return END


def after_tools(state: NutritionState) -> dict:
    """Increment the tool-call round counter after each tool execution."""
    return {"tool_call_rounds": state.get("tool_call_rounds", 0) + 1}


def route_after_tools(state: NutritionState):
    """Conditional edge from after_tools node."""
    if _turn_msg_delta(state) > MAX_TURN_MESSAGES:
        return "summarize"
    return "chatbot"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer):
    """Build and compile the LangGraph StateGraph with the given checkpointer."""
    builder = StateGraph(NutritionState)

    # Nodes
    builder.add_node("load_context", load_context)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_node("after_tools", after_tools)
    builder.add_node("summarize", summarize)

    # Edges
    builder.set_entry_point("load_context")
    builder.add_edge("load_context", "chatbot")
    builder.add_conditional_edges("chatbot", should_continue, {
        "tools": "tools",
        "summarize": "summarize",
        END: END,
    })
    builder.add_edge("tools", "after_tools")
    builder.add_conditional_edges("after_tools", route_after_tools, {
        "chatbot": "chatbot",
        "summarize": "summarize",
    })
    builder.add_edge("summarize", "chatbot")

    return builder.compile(checkpointer=checkpointer)
