"""Channel abstraction — unified interface for multi-platform adapters.

Each chat platform (CLI, WeChat, QQ, Feishu) implements IChannel.
The Gateway server routes incoming messages through SessionManager
to the Agent core, and streams responses back through platform renderers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ChannelType(Enum):
    """Channel type identifier."""

    CLI = auto()
    WECHAT = auto()
    QQ = auto()
    FEISHU = auto()


class MessageType(Enum):
    """Internal message type — platform-normalized."""

    TEXT = auto()
    IMAGE = auto()
    VOICE = auto()
    FILE = auto()
    RICH_TEXT = auto()  # Markdown / rich text blocks (Feishu)
    INTERACTIVE = auto()  # buttons / cards


@dataclass
class IncomingMessage:
    """Normalized incoming message from any platform.

    Attributes:
        channel: Source platform identifier.
        user_id: Platform-specific user unique ID.
        session_id: Logical session name (default: "default").
        content: Plain text body.
        message_type: Kind of message (text, image, etc.).
        raw: Original platform message for debugging/extension.
        reply_to: Message ID this is replying to, if any.
    """

    channel: ChannelType
    user_id: str
    session_id: str = "default"
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    raw: dict[str, Any] = field(default_factory=dict)
    reply_to: str | None = None


@dataclass
class OutgoingMessage:
    """Normalized outgoing message to any platform.

    Platform renderers populate the fields their platform supports.
    Fields that are None are ignored by the platform sender.
    """

    text: str = ""
    rich_text: str | None = None  # Markdown / rich text (Feishu)
    image_url: str | None = None
    file_url: str | None = None
    reply_to: str | None = None
    buttons: list[dict[str, str]] | None = None  # interactive buttons


class IChannel(ABC):
    """Channel plugin interface.

    Lifecycle:
        1. __init__() — load platform configuration.
        2. start(handler) — begin listening for events (webhook / polling / ws).
        3. On message arrival: handler(IncomingMessage) → AsyncIterator[OutgoingMessage].
        4. stop() — graceful shutdown.

    Each platform adapter implements this interface so the Gateway
    can treat all channels uniformly.
    """

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the platform type identifier for this channel."""
        ...

    @abstractmethod
    async def start(
        self,
        handler: Callable[[IncomingMessage], AsyncIterator[OutgoingMessage]],
    ) -> None:
        """Start the event listener.

        When a user message arrives:
            1. Build an IncomingMessage.
            2. Call ``handler(msg)`` to get an AsyncIterator[OutgoingMessage].
            3. Iterate and send each OutgoingMessage to the user.

        This method runs until ``stop()`` is called.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down — disconnect, release resources."""
        ...

    def render(self, event: Any) -> list[Any]:
        """Convert a StreamEvent to platform-sendable messages.

        Default: returns empty list. Subclasses override to produce
        platform-native messages (text segments, cards, etc.).
        """
        return []
