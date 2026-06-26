"""Tests for clawagent.tools."""
# mypy: disallow-untyped-defs = False

from datetime import datetime

import pytest

from clawagent.tools import (
    _resolve_path,
    get_current_time,
    greet,
    read_file,
    run_command,
    write_file,
)


class TestGetCurrentTime:
    def test_returns_iso_format(self) -> None:
        result = get_current_time.invoke({})
        datetime.fromisoformat(result)  # should not raise

    def test_contains_today(self) -> None:
        get_current_time.invoke({})
        assert datetime.now().year in (2025, 2026, 2027)


class TestGreet:
    def test_greets_by_name(self) -> None:
        result = greet.invoke({"name": "Alice"})
        assert "Alice" in result
        assert "Hello" in result

    def test_greets_different_names(self) -> None:
        assert "Bob" in greet.invoke({"name": "Bob"})


class TestReadFile:
    def test_read_existing_file(self) -> None:
        content = read_file.invoke({"path": "README.md"})
        assert "# clawagent" in content

    def test_read_nonexistent_file(self) -> None:
        result = read_file.invoke({"path": "nonexistent_file_xyz.txt"})
        assert "not found" in result.lower() or "no such" in result.lower()

    def test_read_directory(self) -> None:
        result = read_file.invoke({"path": "src"})
        assert "not a file" in result.lower() or "is a directory" in result.lower()

    def test_read_outside_project_fails(self) -> None:
        with pytest.raises(ValueError, match="outside the project"):
            _resolve_path("/etc/passwd")

    def test_read_project_root_allowed(self) -> None:
        resolved = _resolve_path(".")
        assert resolved.exists()


class TestWriteFile:
    def test_write_and_read_back(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import clawagent.tools as t

        original_root = t._PROJECT_ROOT
        try:
            t._PROJECT_ROOT = tmp_path

            result = write_file.invoke({"path": "test.txt", "content": "hello world"})
            assert "11 bytes" in result or "written" in result.lower()

            content = read_file.invoke({"path": "test.txt"})
            assert content == "hello world"
        finally:
            t._PROJECT_ROOT = original_root

    def test_write_outside_project_fails(self) -> None:
        with pytest.raises(ValueError, match="outside the project"):
            write_file.invoke({"path": "/tmp/outside.txt", "content": "test"})

    def test_write_creates_parent_dirs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import clawagent.tools as t
        original = t._PROJECT_ROOT
        try:
            t._PROJECT_ROOT = tmp_path
            write_file.invoke({"path": "a/b/c/deep.txt", "content": "nested"})
            assert (tmp_path / "a/b/c/deep.txt").exists()
            assert "nested" in (tmp_path / "a/b/c/deep.txt").read_text()
        finally:
            t._PROJECT_ROOT = original


class TestRunCommand:
    def test_echo(self) -> None:
        result = run_command.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_pwd(self) -> None:
        result = run_command.invoke({"command": "pwd"})
        assert "clawagent" in result

    def test_failing_command(self) -> None:
        result = run_command.invoke({"command": "exit 1"})
        assert "exit code" in result


class TestResolvePath:
    def test_absolute_path_allowed_if_within(self) -> None:
        resolved = _resolve_path(".")
        assert resolved.is_absolute()

    def test_relative_path(self) -> None:
        resolved = _resolve_path("src")
        assert resolved.is_absolute()
