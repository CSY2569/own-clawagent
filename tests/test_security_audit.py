"""Tests for clawagent.security.audit."""

# mypy: disable-error-code="no-untyped-def"

import json

from clawagent.security.audit import AuditLogger, _sanitize_args


class TestSanitizeArgs:
    def test_short_value_unchanged(self):
        result = _sanitize_args({"path": "src/main.py"})
        assert result == {"path": "src/main.py"}

    def test_long_value_truncated(self):
        long_val = "x" * 300
        result = _sanitize_args({"content": long_val})
        assert len(result["content"]) < 300
        assert "300 chars" in result["content"]

    def test_non_string_unchanged(self):
        result = _sanitize_args({"count": 42, "flag": True})
        assert result == {"count": 42, "flag": True}

    def test_empty(self):
        assert _sanitize_args({}) == {}


class TestAuditLogger:
    def test_write_entry(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_file)
        audit.log("thread-1", "write_file", {"path": "test.txt"}, "confirm", "approved")
        lines = (tmp_path / "audit.jsonl").read_text("utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["thread_id"] == "thread-1"
        assert entry["tool"] == "write_file"
        assert entry["args"] == {"path": "test.txt"}
        assert entry["level"] == "confirm"
        assert entry["result"] == "approved"
        assert "timestamp" in entry

    def test_multiple_entries(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_file)
        for i in range(5):
            audit.log(f"t{i}", "read_file", {"path": f"f{i}"}, "allow", "approved")
        lines = (tmp_path / "audit.jsonl").read_text("utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_creates_parent_dir(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "deep" / "audit.jsonl")
        audit = AuditLogger(log_path=log_file)
        audit.log("t1", "test", {}, "allow", "ok")
        assert (tmp_path / "subdir" / "deep" / "audit.jsonl").exists()

    def test_long_args_sanitized_in_log(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(log_path=log_file)
        long_content = "A" * 500
        audit.log("t1", "write_file", {"content": long_content}, "confirm", "approved")
        entry = json.loads((tmp_path / "audit.jsonl").read_text("utf-8").strip())
        assert "500 chars" in entry["args"]["content"]
        assert len(entry["args"]["content"]) < 500
