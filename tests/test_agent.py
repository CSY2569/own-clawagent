"""Tests for clawagent.agent."""

from clawagent.agent import AgentResponse, Usage, _extract_text


class TestUsage:
    def test_defaults(self) -> None:
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_from_response_metadata(self) -> None:
        metadata = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 10,
            }
        }
        u = Usage.from_response_metadata(metadata)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.cache_read_input_tokens == 20

    def test_from_empty_metadata(self) -> None:
        u = Usage.from_response_metadata({})
        assert u == Usage()

    def test_from_metadata_missing_usage_key(self) -> None:
        u = Usage.from_response_metadata({"other": 1})
        assert u == Usage()


class TestAgentResponse:
    def test_create(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=5)
        resp = AgentResponse(text="Hello", usage=usage)
        assert resp.text == "Hello"
        assert resp.usage.input_tokens == 10


class TestExtractText:
    def test_plain_string(self) -> None:
        assert _extract_text("hello") == "hello"

    def test_list_of_content_blocks(self) -> None:
        blocks = [
            {"type": "thinking", "text": "hidden"},
            {"type": "text", "text": "visible"},
        ]
        assert _extract_text(blocks) == "visible"

    def test_multiple_text_blocks(self) -> None:
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert _extract_text(blocks) == "hello\nworld"

    def test_empty_list(self) -> None:
        assert _extract_text([]) == ""

    def test_none(self) -> None:
        assert _extract_text(None) == "None"

    def test_no_text_blocks(self) -> None:
        blocks = [{"type": "thinking", "text": "hidden"}]
        assert _extract_text(blocks) == ""
