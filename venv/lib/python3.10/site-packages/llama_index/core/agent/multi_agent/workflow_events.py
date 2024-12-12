from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event

class ToolApprovalNeeded(Event):
    """Emitted when a tool call needs approval."""
    id: str
    tool_name: str
    tool_kwargs: dict

class ApproveTool(Event):
    """Required to approve a tool."""
    id: str
    tool_name: str
    tool_kwargs: dict
    approved: bool

class LLMEvent(Event):
    """All LLM calls are shown."""
    input: list[ChatMessage]
    delta: str
    response: str
    raw_response: Any

class ToolCall(Event):
    """All tool calls are surfaced."""
    tool_name: str
    tool_kwargs: dict
    tool_output: Any

class HandoffEvent(Event):
    """Internal event for agent handoffs."""
    from_agent: str
    to_agent: str
    reason: str
