"""Tests for clawagent.security.permissions."""

# mypy: disable-error-code="no-untyped-def"

from clawagent.security.permissions import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
    _extract_pattern,
)


class TestPermissionLevel:
    def test_values(self):
        assert PermissionLevel.ALLOW.value == "allow"
        assert PermissionLevel.CONFIRM.value == "confirm"
        assert PermissionLevel.DENY.value == "deny"


class TestExtractPattern:
    def test_write_file_path(self):
        assert _extract_pattern("write_file", {"path": "src/main.py"}) == "src/main.py"

    def test_write_file_arg1(self):
        assert _extract_pattern("write_file", {"arg1": "output/result.txt"}) == "output/result.txt"

    def test_read_file_path(self):
        assert _extract_pattern("read_file", {"path": "README.md"}) == "README.md"

    def test_run_command_first_word(self):
        assert _extract_pattern("run_command", {"command": "git status"}) == "git"

    def test_run_command_empty(self):
        assert _extract_pattern("run_command", {"command": ""}) == ""

    def test_unknown_tool_returns_star(self):
        assert _extract_pattern("unknown_tool", {"foo": "bar"}) == "*"


class TestPermissionConfigMatch:
    def test_deny_rule_matches(self):
        config = PermissionConfig()
        rule = config.match("run_command", {"command": "rm -rf /"})
        assert rule.level == PermissionLevel.DENY

    def test_allow_rule_matches(self):
        config = PermissionConfig()
        rule = config.match("read_file", {"path": "README.md"})
        assert rule.level == PermissionLevel.ALLOW

    def test_confirm_rule_matches(self):
        config = PermissionConfig()
        rule = config.match("write_file", {"path": "src/main.py"})
        assert rule.level == PermissionLevel.CONFIRM

    def test_fallback_to_default_allow(self):
        config = PermissionConfig()
        rule = config.match("nonexistent_tool", {"arg1": "x"})
        assert rule.level == PermissionLevel.ALLOW

    def test_glob_pattern_match(self):
        rule = PermissionRule("write_file", "*.env*", PermissionLevel.DENY)
        config = PermissionConfig(rules=[rule])
        assert config.match("write_file", {"path": ".env"}).level == PermissionLevel.DENY
        assert config.match("write_file", {"path": ".env.local"}).level == PermissionLevel.DENY
        assert config.match("write_file", {"path": "config.env.bak"}).level == PermissionLevel.DENY

    def test_first_match_wins(self):
        rule1 = PermissionRule("write_file", "*", PermissionLevel.DENY)
        rule2 = PermissionRule("write_file", "output/*", PermissionLevel.ALLOW)
        config = PermissionConfig(rules=[rule1, rule2])
        # First rule (deny all) should win even for output paths
        assert config.match("write_file", {"path": "output/test.txt"}).level == PermissionLevel.DENY


class TestAddRule:
    def test_add_rule_takes_priority(self):
        config = PermissionConfig()
        # Default: write_file to any path is CONFIRM
        assert config.match("write_file", {"path": "test.txt"}).level == PermissionLevel.CONFIRM
        # Add a deny rule at the front
        config.add_rule("write_file", "test.txt", PermissionLevel.DENY, "blocked")
        assert config.match("write_file", {"path": "test.txt"}).level == PermissionLevel.DENY

    def test_reset(self):
        config = PermissionConfig()
        config.add_rule("read_file", "*", PermissionLevel.DENY)
        assert config.match("read_file", {"path": "x"}).level == PermissionLevel.DENY
        config.reset()
        assert config.match("read_file", {"path": "x"}).level == PermissionLevel.ALLOW


class TestForWorker:
    def test_researcher_denied_write(self):
        config = PermissionConfig.for_worker("researcher")
        assert config.match("write_file", {"path": "any.txt"}).level == PermissionLevel.DENY

    def test_researcher_denied_command(self):
        config = PermissionConfig.for_worker("researcher")
        assert config.match("run_command", {"command": "ls"}).level == PermissionLevel.DENY

    def test_critic_denied_write(self):
        config = PermissionConfig.for_worker("critic")
        assert config.match("write_file", {"path": "any.txt"}).level == PermissionLevel.DENY

    def test_writer_denied_command(self):
        config = PermissionConfig.for_worker("writer")
        assert config.match("run_command", {"command": "ls"}).level == PermissionLevel.DENY

    def test_coder_no_extra_restrictions(self):
        config = PermissionConfig.for_worker("coder")
        # Coder can still write (with confirm from default rules)
        assert config.match("write_file", {"path": "output/x"}).level == PermissionLevel.ALLOW

    def test_unknown_role_no_restrictions(self):
        config = PermissionConfig.for_worker("unknown_role")
        assert config.match("write_file", {"path": "output/x"}).level == PermissionLevel.ALLOW


class TestDefaultRules:
    def test_env_files_denied(self):
        config = PermissionConfig()
        assert config.match("write_file", {"path": ".env"}).level == PermissionLevel.DENY
        assert config.match("write_file", {"path": ".env.local"}).level == PermissionLevel.DENY

    def test_output_dir_allowed(self):
        config = PermissionConfig()
        assert config.match("write_file", {"path": "output/result.txt"}).level == PermissionLevel.ALLOW

    def test_rm_denied(self):
        config = PermissionConfig()
        assert config.match("run_command", {"command": "rm -rf /"}).level == PermissionLevel.DENY

    def test_git_read_allowed(self):
        config = PermissionConfig()
        assert config.match("run_command", {"command": "git status"}).level == PermissionLevel.ALLOW
        assert config.match("run_command", {"command": "git log"}).level == PermissionLevel.ALLOW
