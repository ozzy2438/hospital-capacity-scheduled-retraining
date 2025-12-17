"""Basic unit tests for the promotion gating behavior.

These tests validate the core policy:
- If no champion exists, challenger is eligible for promotion.
- If a champion exists, challenger must clear a relative improvement threshold.

The DAG code is purposely simple and local-registry-based; these tests exercise
the underlying metric math rather than Airflow execution."""

import math


def should_promote(mae_challenger: float, mae_champion: float | None, min_rel_improve: float) -> bool:
    if mae_champion is None:
        return True
    return mae_challenger <= (1.0 - min_rel_improve) * mae_champion


def test_promote_when_no_champion() -> None:
    assert should_promote(mae_challenger=10.0, mae_champion=None, min_rel_improve=0.02) is True


def test_promote_when_improves_enough() -> None:
    # 2% better than champion
    assert should_promote(mae_challenger=9.8, mae_champion=10.0, min_rel_improve=0.02) is True


def test_do_not_promote_when_not_enough_improvement() -> None:
    assert should_promote(mae_challenger=9.9, mae_champion=10.0, min_rel_improve=0.02) is False


def test_handles_float_edges() -> None:
    # Very small threshold; challenger must still be <= scaled champion
    assert should_promote(mae_challenger=1.0000001, mae_champion=1.0, min_rel_improve=0.0) is False
    assert should_promote(mae_challenger=1.0, mae_champion=1.0, min_rel_improve=0.0) is True
    assert math.isclose((1.0 - 0.02) * 10.0, 9.8)
