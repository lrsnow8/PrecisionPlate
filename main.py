import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from db.database import bootstrap_user
from agent.graph import build_graph

console = Console()


def main():
    # Ensure API key is present
    if not os.environ.get("CLAUDE_API_KEY"):
        console.print("[bold red]Error:[/bold red] CLAUDE_API_KEY environment variable is not set.")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold green]PrecisionPlate[/bold green] — Your personal nutrition assistant\n"
        "Type your message and press Enter. Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
        border_style="green",
    ))

    # Bootstrap user — always returns "1"
    user_id = bootstrap_user()
    thread_id = user_id
    config = {"configurable": {"thread_id": thread_id}}

    session = PromptSession(history=InMemoryHistory())

    # SqliteSaver.from_conn_string() is a context manager in LangGraph 1.x
    with SqliteSaver.from_conn_string("db/precision_plate.db") as checkpointer:
        graph = build_graph(checkpointer)

        while True:
            try:
                user_input = session.prompt("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "user_id": user_id,
                    },
                    config=config,
                )

                # Extract the last AI message
                messages = result.get("messages", [])
                ai_reply = None
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "ai":
                        ai_reply = msg.content
                        break

                if ai_reply:
                    console.print("\n[bold cyan]PrecisionPlate:[/bold cyan]")
                    console.print(Markdown(ai_reply))
                else:
                    console.print("[dim]No response.[/dim]")

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
