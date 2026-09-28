import pytest

from kasoti.config import settings
from kasoti.metrics import classification_report, compute_metrics
from kasoti.models import PredictionRecord


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_classification_report_matches_hand_computed_fixture():
    # 8 hand-picked (true, pred) category pairs — worked out by hand below.
    y_true = ["billing", "billing", "bug", "bug", "other", "other", "billing", "bug"]
    y_pred = ["billing", "bug", "bug", "other", "other", "other", "billing", "bug"]

    report = classification_report(["billing", "bug", "other"], y_true, y_pred)

    assert report["confusion_matrix"] == {
        "billing": {"billing": 2, "bug": 1, "other": 0},
        "bug": {"billing": 0, "bug": 2, "other": 1},
        "other": {"billing": 0, "bug": 0, "other": 2},
    }
    # 6 of 8 correct: both mistakes are billing->bug and bug->other.
    assert report["accuracy"] == pytest.approx(6 / 8)

    # billing: tp=2, fp=0, fn=1 (one billing ticket predicted as bug)
    assert report["per_class"]["billing"]["precision"] == pytest.approx(1.0)
    assert report["per_class"]["billing"]["recall"] == pytest.approx(2 / 3)
    assert report["per_class"]["billing"]["f1"] == pytest.approx(0.8)

    # bug: tp=2, fp=1 (billing predicted as bug), fn=1 (bug predicted as other)
    assert report["per_class"]["bug"]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"]["bug"]["recall"] == pytest.approx(2 / 3)

    # other: tp=2, fp=1 (bug predicted as other), fn=0
    assert report["per_class"]["other"]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"]["other"]["recall"] == pytest.approx(1.0)

    assert report["macro"]["precision"] == pytest.approx(_mean([1.0, 2 / 3, 2 / 3]))


def _record(**overrides) -> PredictionRecord:
    base = {
        "id": "t1",
        "true_category": "billing",
        "true_urgency": "low",
        "true_needs_human": False,
        "pred_category": "billing",
        "pred_urgency": "low",
        "pred_needs_human": False,
        "source": "rule",
        "latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "schema_valid": True,
    }
    base.update(overrides)
    return PredictionRecord(**base)


def test_compute_metrics_hit_rate_latency_and_cost():
    records = [
        _record(id="1", source="rule", rule_id="billing_refund_request", confidence=0.9, latency_ms=0.0),
        _record(
            id="2",
            true_category="bug",
            true_urgency="high",
            true_needs_human=True,
            pred_category="bug",
            pred_urgency="high",
            pred_needs_human=True,
            source="api",
            latency_ms=100.0,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            schema_valid=True,
        ),
        _record(
            id="3",
            true_category="other",
            true_urgency="low",
            true_needs_human=False,
            pred_category="other",
            pred_urgency="low",
            pred_needs_human=False,
            source="api",
            latency_ms=200.0,
            input_tokens=0,
            output_tokens=0,
            schema_valid=False,
        ),
    ]

    metrics = compute_metrics(records)

    assert metrics["total_tickets"] == 3
    assert metrics["rule_hit_rate"] == pytest.approx(1 / 3)
    assert metrics["api_calls_made"] == 2
    assert metrics["api_calls_saved"] == 1
    assert metrics["schema_violation_rate"] == pytest.approx(1 / 2)
    assert metrics["latency_ms"]["p50"] == pytest.approx(150.0)
    assert metrics["latency_ms"]["p95"] == pytest.approx(195.0)

    expected_cost_usd = (
        1_000_000 / 1_000_000 * settings.input_price_per_mtok_usd
        + 1_000_000 / 1_000_000 * settings.output_price_per_mtok_usd
    )
    expected_cost_inr_per_1000 = expected_cost_usd * settings.usd_to_inr / 3 * 1000
    assert metrics["cost_inr_per_1000_tickets"] == pytest.approx(expected_cost_inr_per_1000)
