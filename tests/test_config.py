"""Tests for clawagent.config."""
# mypy: disallow-untyped-defs = False

from pathlib import Path

from pytest import MonkeyPatch

from clawagent.config import PriceBook, PriceConfig, Settings, _extract_price


class TestSettings:
    def test_from_env_with_key(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test-key")
        s = Settings.from_env()
        assert s.api_key == "sk-test-key"
        assert s.model_name == "deepseek-v4-flash"
        assert s.context_window == 1_000_000

    def test_from_env_missing_key(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("CLAWAGENT_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        try:
            Settings.from_env()
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_custom_model_and_window(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        monkeypatch.setenv("CLAWAGENT_MODEL", "deepseek-v4-pro")
        monkeypatch.setenv("CLAWAGENT_CONTEXT_WINDOW", "128000")
        s = Settings.from_env()
        assert s.model_name == "deepseek-v4-pro"
        assert s.context_window == 128000

    def test_invalid_context_window_fallback(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        monkeypatch.setenv("CLAWAGENT_CONTEXT_WINDOW", "invalid")
        s = Settings.from_env()
        assert s.context_window == 1_000_000


class TestPriceConfig:
    def test_defaults(self) -> None:
        pc = PriceConfig()
        assert pc.input_per_1m == 0.0
        assert pc.cache_hit_per_1m == 0.0
        assert pc.output_per_1m == 0.0

    def test_custom_values(self) -> None:
        pc = PriceConfig(input_per_1m=1.0, cache_hit_per_1m=0.02, output_per_1m=2.0)
        assert pc.input_per_1m == 1.0
        assert pc.cache_hit_per_1m == 0.02


class TestPriceBook:
    def test_empty_book(self) -> None:
        pb = PriceBook()
        assert pb.get("anything") == PriceConfig()

    def test_get_known_model(self) -> None:
        pb = PriceBook(models={"test-model": PriceConfig(input_per_1m=5.0)})
        pc = pb.get("test-model")
        assert pc.input_per_1m == 5.0

    def test_get_unknown_model(self) -> None:
        pb = PriceBook(models={"model-a": PriceConfig()})
        assert pb.get("model-b") == PriceConfig()


class TestExtractPrice:
    def test_extract_single(self) -> None:
        text = "价格 1.5元 2.0元"
        result = _extract_price(text, r"([\d.]+)元")
        assert result == [1.5, 2.0]

    def test_extract_empty(self) -> None:
        assert _extract_price("no prices", r"([\d.]+)元") == []


class TestLoadPriceBook:
    def test_missing_file(self, tmp_path: Path) -> None:
        import clawagent.config as cfg

        pb = cfg.load_price_book(tmp_path / "nonexistent.txt")
        assert pb == PriceBook()

    def test_parse_table_format(self, tmp_path: Path) -> None:
        price_file = tmp_path / "price.txt"
        price_file.write_text(
            "模型  deepseek-v4-flash  deepseek-v4-pro\n"
            "百万tokens输入(缓存命中)  0.02元  0.025元\n"
            "百万tokens输入(缓存未命中)  1元  3元\n"
            "百万tokens输出  2元  6元\n",
            encoding="utf-8",
        )

        import clawagent.config as cfg

        pb = cfg.load_price_book(price_file)

        flash = pb.get("deepseek-v4-flash")
        assert flash.input_per_1m == 1.0
        assert flash.cache_hit_per_1m == 0.02
        assert flash.output_per_1m == 2.0

        pro = pb.get("deepseek-v4-pro")
        assert pro.input_per_1m == 3.0
        assert pro.cache_hit_per_1m == 0.025
        assert pro.output_per_1m == 6.0
