from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class NutritionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation (managed by checkpointer)
    summary: str          # compressed long-term memory summary
    user_id: str          # identifies the user across sessions
    today_snapshot: dict  # today's macro totals, refreshed each turn
    tool_call_rounds: int  # counts consecutive chatbot→tools cycles this turn
    turn_start_msg_count: int  # message count at the start of the current user turn
