"""Tests for concrete worker implementations — tool sets and prompt customization."""

# mypy: disable-error-code="no-untyped-def"

from clawagent.worker.coder import CoderWorker
from clawagent.worker.config import WorkerConfig
from clawagent.worker.critic import CriticWorker
from clawagent.worker.researcher import ResearcherWorker
from clawagent.worker.writer import WriterWorker


def _make_config(role: str) -> WorkerConfig:
    return WorkerConfig(role=role)


class TestCoderWorker:
    def test_tools(self):
        w = CoderWorker(_make_config("coder"))
        names = {t.name for t in w._get_tools()}
        assert names == {"read_file", "write_file", "run_command"}

    def test_no_search_tools(self):
        """Coder 没有搜索相关工具。"""
        w = CoderWorker(_make_config("coder"))
        names = {t.name for t in w._get_tools()}
        assert "search_documents" not in names

    def test_default_prompt_includes_task(self):
        """Coder 使用默认 _customize_prompt，末尾包含 Current Task。"""
        w = CoderWorker(_make_config("coder"))
        prompt = w.build_prompt("write a function")
        assert "## Current Task" in prompt
        assert "write a function" in prompt


class TestResearcherWorker:
    def test_tools(self):
        w = ResearcherWorker(_make_config("researcher"))
        names = {t.name for t in w._get_tools()}
        assert names == {"search_documents", "web_search"}

    def test_no_write_tools(self):
        """Researcher 没有写文件或执行命令的工具。"""
        w = ResearcherWorker(_make_config("researcher"))
        names = {t.name for t in w._get_tools()}
        assert "write_file" not in names
        assert "run_command" not in names


class TestCriticWorker:
    def test_tools(self):
        w = CriticWorker(_make_config("critic"))
        names = {t.name for t in w._get_tools()}
        assert names == {"read_file", "search_documents"}

    def test_custom_prompt_format(self, tmp_path):
        """提供 task-template.md 时使用模板文件。"""
        role_dir = tmp_path / "agents" / "critic"
        role_dir.mkdir(parents=True)
        (role_dir / "task-template.md").write_text(
            "## 审查任务\n{task}\n\n## 问题列表\n| 严重程度 | 建议修复 |\n")
        w = CriticWorker(_make_config("critic"))
        prompt = w._customize_prompt("base prompt", "review this code", str(tmp_path))
        assert "## 问题列表" in prompt
        assert "严重程度" in prompt
        assert "建议修复" in prompt
        assert "review this code" in prompt

    def test_custom_prompt_fallback(self):
        """无 task-template.md 时回退到默认 ## Current Task 格式。"""
        w = CriticWorker(_make_config("critic"))
        prompt = w._customize_prompt("base", "review")
        assert "## Current Task" in prompt
        assert "review" in prompt

    def test_no_write_tools(self):
        """Critic 只读，没有写工具。"""
        w = CriticWorker(_make_config("critic"))
        names = {t.name for t in w._get_tools()}
        assert "write_file" not in names
        assert "run_command" not in names


class TestWriterWorker:
    def test_tools(self):
        w = WriterWorker(_make_config("writer"))
        names = {t.name for t in w._get_tools()}
        assert names == {"read_file", "write_file"}

    def test_no_run_tools(self):
        """Writer 没有执行命令和搜索的工具。"""
        w = WriterWorker(_make_config("writer"))
        names = {t.name for t in w._get_tools()}
        assert "run_command" not in names
        assert "search_documents" not in names
