"""Tests for clawagent.ui."""

import time

from clawagent.agent import Usage
from clawagent.config import PriceConfig
from clawagent.ui import (
    ConversationStats,
    _context_color,
    _format_cost,
    _format_duration,
    _format_tokens,
)


class TestFormatDuration:
    def test_zero(self) -> None:
        assert _format_duration(0) == "0s"

    def test_seconds(self) -> None:
        assert _format_duration(45) == "45s"

    def test_minutes(self) -> None:
        assert _format_duration(125) == "2m05s"

    def test_exact_minute(self) -> None:
        assert _format_duration(60) == "1m00s"

    def test_negative(self) -> None:
        assert _format_duration(-5) == "0s"


class TestFormatTokens:
    def test_under_thousand(self) -> None:
        assert _format_tokens(500) == "500"

    def test_thousands(self) -> None:
        assert _format_tokens(1500) == "1.5K"

    def test_millions(self) -> None:
        assert _format_tokens(3_500_000) == "3.5M"

    def test_zero(self) -> None:
        assert _format_tokens(0) == "0"


class TestFormatCost:
    def test_zero(self) -> None:
        assert _format_cost(0) == "¥0.00"

    def test_small_amount(self) -> None:
        assert _format_cost(0.005) == "¥0.00"

    def test_normal_amount(self) -> None:
        result = _format_cost(1.5)
        assert result.startswith("¥")
        assert "1.5" in result

    def test_exact_three_decimals(self) -> None:
        result = _format_cost(0.123)
        assert "¥" in result
        # Should strip trailing zeros after truncation to 4 decimals
        assert len(result) > 2


class TestContextColor:
    def test_green_below_70(self) -> None:
        assert _context_color(0) == "green"
        assert _context_color(69) == "green"

    def test_yellow_70_to_90(self) -> None:
        assert _context_color(70) == "yellow"
        assert _context_color(89) == "yellow"

    def test_red_90_and_above(self) -> None:
        assert _context_color(90) == "red"
        assert _context_color(100) == "red"
        assert _context_color(200) == "red"


class TestConversationStats:
    def test_initial_state(self) -> None:
        cs = ConversationStats()
        assert cs.total_tokens == 0
        assert cs.message_count == 0
        assert cs.context_usage_pct(1_000_000) == 0.0
        assert cs.cost(PriceConfig()) == 0.0

    def test_update_increments(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=100, output_tokens=50))
        assert cs.cumulative_input_tokens == 100
        assert cs.cumulative_output_tokens == 50
        assert cs.message_count == 1
        assert cs.latest_input_tokens == 100

    def test_update_accumulates(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=100, output_tokens=50))
        cs.update(Usage(input_tokens=200, output_tokens=30))
        assert cs.cumulative_input_tokens == 300
        assert cs.cumulative_output_tokens == 80
        assert cs.message_count == 2
        assert cs.latest_input_tokens == 200

    def test_total_tokens(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=100, output_tokens=50))
        assert cs.total_tokens == 150

    def test_context_usage_pct(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=50_000))
        assert cs.context_usage_pct(1_000_000) == 5.0

    def test_context_usage_zero_window(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=100))
        assert cs.context_usage_pct(0) == 0.0

    def test_cost_calculation(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=1_000_000, output_tokens=500_000))
        pricing = PriceConfig(input_per_1m=1.0, cache_hit_per_1m=0.02, output_per_1m=2.0)
        assert cs.cost(pricing) == 2.0

    def test_cost_with_cache_hit(self) -> None:
        cs = ConversationStats()
        cs.update(Usage(input_tokens=1_000_000, cache_read_input_tokens=500_000))
        pricing = PriceConfig(input_per_1m=1.0, cache_hit_per_1m=0.02, output_per_1m=2.0)
        expected = 1.0 + 0.01  # 1.0 (miss) + 0.01 (hit)
        assert cs.cost(pricing) == expected

    def test_elapsed_seconds_monotonic(self) -> None:
        cs = ConversationStats(start_time=time.monotonic())
        time.sleep(0.01)
        assert cs.elapsed_seconds > 0

    def test_elapsed_zero_when_no_start_time(self) -> None:
        cs = ConversationStats()
        assert cs.elapsed_seconds == 0.0
