"""Token estimation utilities for context compression."""

from langchain_core.messages import BaseMessage

from clawagent.utils import extract_text


def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Estimate total token count for a list of messages.

    Estimation formula (conservative):
    - ASCII characters: 4 chars = 1 token
    - Chinese/non-ASCII: 2 chars = 1 token
    - +4 tokens per message for metadata overhead (role, etc.)

    This is a conservative estimate; actual token counts are usually lower.
    For precise counts, use tiktoken with the specific model's tokenizer.
    """
    total = 0
    for msg in messages:
        content = extract_text(msg.content)
        ascii_count = sum(1 for c in content if ord(c) < 128)
        non_ascii = len(content) - ascii_count

        tokens = ascii_count // 4 + non_ascii // 2 + 4
        total += tokens

    return total
