"""Interactive chat loop with Claude and transparent memory operations."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from chicory.cli.commands import handle_slash_command
from chicory.cli.display import console
from chicory.config import ChicoryConfig
from chicory.llm.client import ClaudeClient
from chicory.orchestrator.orchestrator import Orchestrator
from chicory.orchestrator.tool_handlers import dispatch_tool_call


class ChatSession:
    """Interactive chat session with Claude, memory ops happening transparently."""

    def __init__(self, config: ChicoryConfig) -> None:
        self._config = config
        self._orchestrator = Orchestrator(config)
        self._client = ClaudeClient(config)
        self._messages: list[dict[str, Any]] = []

    def run(self) -> None:
        """Run the interactive chat loop."""
        console.print()
        console.print("[bold green]Chicory[/bold green] — Memory-augmented conversation")
        console.print("[dim]Type /help for commands, /quit to exit[/dim]")
        console.print()

        try:
            while True:
                try:
                    user_input = console.input("[bold blue]You:[/bold blue] ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("/quit", "/exit"):
                    break

                if user_input.startswith("/"):
                    if handle_slash_command(self._orchestrator, user_input):
                        continue
                    console.print(f"[dim]Unknown command: {user_input.split()[0]}[/dim]")
                    continue

                self._process_message(user_input)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        finally:
            self._orchestrator.close()
            console.print("[dim]Session ended.[/dim]")

    def _process_message(self, user_input: str) -> None:
        """Process a user message through Claude with tool handling."""
        # Update active tags in system prompt
        active_tags = self._orchestrator.tag_manager.list_active_names()
        self._client.update_active_tags(active_tags)

        self._messages.append({"role": "user", "content": user_input})

        # Send to Claude and handle tool use loop
        empty_retries = 0
        while True:
            try:
                response = self._client.chat(self._messages)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                self._cleanup_orphaned_tool_use()
                return

            # Process response blocks
            assistant_content = response.content
            has_tool_use = any(
                block.type == "tool_use" for block in assistant_content
            )

            if not has_tool_use:
                # Check for empty or truncated responses
                has_text = any(
                    hasattr(b, "text") and b.text.strip()
                    for b in assistant_content
                )
                if not has_text and hasattr(response, "stop_reason"):
                    # Debug: show what blocks we actually got
                    block_types = [b.type for b in assistant_content]
                    if block_types:
                        console.print(
                            f"[dim](Response contained: {block_types}, "
                            f"stop_reason={response.stop_reason})[/dim]"
                        )
                    if response.stop_reason == "max_tokens":
                        console.print(
                            "[yellow]Response was cut off (max tokens). "
                            "Continuing...[/yellow]"
                        )
                        self._messages.append({
                            "role": "assistant",
                            "content": assistant_content,
                        })
                        self._messages.append({
                            "role": "user",
                            "content": "Please continue your response.",
                        })
                        continue
                    elif empty_retries < 2:
                        empty_retries += 1
                        console.print(
                            "[dim](No response text received — retrying...)[/dim]"
                        )
                        continue
                    else:
                        console.print(
                            "[dim](No response text after retries.)[/dim]"
                        )
                        return

                # Final text response
                self._messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })
                for block in assistant_content:
                    if hasattr(block, "text") and block.text.strip():
                        console.print()
                        console.print(Markdown(block.text))
                        console.print()
                return

            # Handle tool calls
            self._messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    console.print(
                        f"  [dim]Using {block.name}...[/dim]"
                    )
                    try:
                        result = dispatch_tool_call(
                            self._orchestrator, block.name, block.input
                        )
                    except Exception as e:
                        result = {"error": str(e)}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

            self._messages.append({
                "role": "user",
                "content": tool_results,
            })
            # Loop continues — Claude will process tool results

    def _cleanup_orphaned_tool_use(self) -> None:
        """Remove trailing messages that would leave orphaned tool_use blocks.

        After an API error, the message history may end with:
          [..., assistant(tool_use), user(tool_results), ...]
        or just:
          [..., assistant(tool_use)]
        Either way, we need to strip back to the last complete exchange
        so the next API call won't get a 400 error about unmatched blocks.
        """
        while self._messages:
            last = self._messages[-1]
            # If it ends with the user's original text message, remove it
            # (the failed request's input)
            if last["role"] == "user" and isinstance(last["content"], str):
                self._messages.pop()
                break
            # Remove any trailing tool_result or tool_use messages
            self._messages.pop()
