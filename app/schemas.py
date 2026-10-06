"""Shared data structures and the strict memory-change schema."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Category(str, Enum):
    state = "state"
    objective = "objective"
    constraint = "constraint"
    decision = "decision"
    lesson = "lesson"
    discovery = "discovery"
    environment = "environment"


class Operation(str, Enum):
    add = "add"
    update = "update"
    deactivate = "deactivate"


# category -> (file name, id prefix, heading, label for the inactive section)
CATEGORY_INFO: dict[Category, tuple[str, str, str, str]] = {
    Category.state: ("STATE.md", "S", "STATE", "Superseded"),
    Category.objective: ("OBJECTIVES.md", "O", "OBJECTIVES", "Completed / Inactive"),
    Category.constraint: ("CONSTRAINTS.md", "C", "CONSTRAINTS", "Inactive"),
    Category.decision: ("DECISIONS.md", "D", "DECISIONS", "Superseded"),
    Category.lesson: ("LESSONS.md", "L", "LESSONS", "Inactive"),
    Category.discovery: ("DISCOVERIES.md", "DS", "DISCOVERIES", "Inactive"),
    Category.environment: ("ENVIRONMENT.md", "E", "ENVIRONMENT", "Superseded"),
}

ENTRY_ID_RE = re.compile(r"^(S|O|C|D|L|DS|E)-\d{3,6}$")


def prefix_for(category: Category) -> str:
    return CATEGORY_INFO[category][1]


class MemoryChange(BaseModel):
    """One proposed change, exactly as the memory model must emit it.

    extra="forbid": the model cannot smuggle in paths, commands or other fields.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Category
    operation: Operation
    title: str = Field(min_length=1, max_length=120)
    content: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field("", max_length=500)
    target_id: str | None = Field(None, max_length=10)


class MemoryChangeSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changes: list[Any] = Field(default_factory=list, max_length=20)


# JSON schema handed to Ollama's structured-output `format` parameter.
MEMORY_CHANGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": [c.value for c in Category]},
                    "operation": {"type": "string", "enum": [o.value for o in Operation]},
                    "target_id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["category", "operation", "title", "content", "confidence", "reason"],
            },
        }
    },
    "required": ["changes"],
}


@dataclass
class MemoryEntry:
    entry_id: str
    category: Category
    title: str
    content: str
    active: bool = True
    created_at: str = ""
    updated_at: str = ""
    source: str = ""

    def render_line(self) -> str:
        return f"[{self.entry_id}] {self.title}: {self.content}"


@dataclass
class ToolEvent:
    type: str  # "tool_call" | "tool_result"
    tool: str
    arguments: Any = None
    result: Any = None


@dataclass
class InteractionTask:
    conversation_id: str
    timestamp: str
    user_message: str
    assistant_response: str
    tool_events: list[dict] = field(default_factory=list)
    project_id: str = "default"
    source: str = "chat"
    flags: list[dict] = field(default_factory=list)   # [{"category": ..., "text": ...}]
    turn_number: int = 0                              # 1-based user turn within the conversation

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "tool_events": self.tool_events,
            "project_id": self.project_id,
            "source": self.source,
            "flags": self.flags,
            "turn_number": self.turn_number,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InteractionTask":
        return cls(
            conversation_id=d["conversation_id"],
            timestamp=d["timestamp"],
            user_message=d.get("user_message", ""),
            assistant_response=d.get("assistant_response", ""),
            tool_events=list(d.get("tool_events") or []),
            project_id=d.get("project_id", "default"),
            source=d.get("source", "chat"),
            flags=list(d.get("flags") or []),
            turn_number=int(d.get("turn_number") or 0),
        )
