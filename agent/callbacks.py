"""
Rich-styled LangChain callback handler for live execution tracing.

Usage:
    from agent.callbacks import RichCallbackHandler
    handler = RichCallbackHandler()
    graph.invoke(..., config={"callbacks": [handler]})
"""

import json
from typing import Any, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

console = Console(stderr=True)

_DIM    = "dim"
_YELLOW = "bold yellow"
_BLUE   = "bold blue"
_GREEN  = "bold green"
_RED    = "bold red"
_CYAN   = "cyan"


def _short(value: Any, max_len: int = 200) -> str:
    """Return a truncated string representation of value."""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= max_len else s[:max_len] + "…"


class RichCallbackHandler(BaseCallbackHandler):
    """Prints colour-coded traces to stderr while the agent runs."""

    # ------------------------------------------------------------------ LLM

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = serialized.get("kwargs", {}).get("model", "llm")
        flat = [m for batch in messages for m in batch]
        console.print(Rule(f"[{_YELLOW}]LLM call → {model}[/{_YELLOW}]", style=_DIM))
        for msg in flat:
            role = getattr(msg, "type", "?")
            raw = msg.content
            if isinstance(raw, list):
                text = " ".join(
                    b.get("text", "") for b in raw
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip() or "<tool blocks>"
            else:
                text = raw
            console.print(
                Text.assemble(
                    (f"  [{role}] ", _DIM),
                    (_short(text, 120), _CYAN),
                )
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Show token usage when available
        usage = {}
        if response.llm_output:
            usage = response.llm_output.get("usage", {})
        if usage:
            console.print(
                Text.assemble(
                    ("  tokens — ", _DIM),
                    (f"in:{usage.get('input_tokens','?')} ", _DIM),
                    (f"out:{usage.get('output_tokens','?')}", _DIM),
                )
            )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        console.print(f"[{_RED}]  LLM error:[/{_RED}] {error}")

    # ------------------------------------------------------------------ Tools

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "unknown_tool")
        console.print(
            Text.assemble(
                (f"  TOOL ", _YELLOW),
                (f"{name}", _YELLOW),
                (" → input: ", _DIM),
                (_short(input_str, 160), _CYAN),
            )
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        out = output if isinstance(output, str) else str(output)
        console.print(
            Text.assemble(
                ("  TOOL result: ", _DIM),
                (_short(out, 160), _GREEN),
            )
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        console.print(f"[{_RED}]  TOOL error:[/{_RED}] {error}")

    # ------------------------------------------------------------------ Chain

    def on_chain_start(
        self,
        serialized: dict | None,
        inputs: dict,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not serialized:
            return
        name = serialized.get("id", ["?"])[-1]
        # Skip noisy low-level chain names
        if name in ("RunnableSequence", "RunnableLambda", "RunnableParallel"):
            return
        console.print(
            Text.assemble(
                ("  CHAIN ", _BLUE),
                (name, _BLUE),
                (" started", _DIM),
            )
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        console.print(f"[{_RED}]  CHAIN error:[/{_RED}] {error}")
