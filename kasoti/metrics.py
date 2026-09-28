"""Pure metrics functions: precision/recall/F1, confusion matrix, rule hit
rate, cost, latency percentiles, and schema violation rate. Everything here
takes a list of PredictionRecord and returns plain JSON-serializable dicts,
which is what makes it straightforward to unit-test against a hand-computed
fixture.
"""

from __future__ import annotations

import math
from statistics import mean

from kasoti.config import settings
from kasoti.models import CATEGORIES, URGENCIES, PredictionRecord


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def classification_report(labels: list[str], y_true: list[str], y_pred: list[str]) -> dict:
    """Confusion matrix + per-class and macro-averaged precision/recall/F1."""
    confusion = {t: dict.fromkeys(labels, 0) for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t][p] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        support = sum(confusion[label].values())
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(y_true)
    accuracy = sum(confusion[label][label] for label in labels) / total if total else 0.0
    macro = {
        "precision": mean(per_class[label]["precision"] for label in labels) if labels else 0.0,
        "recall": mean(per_class[label]["recall"] for label in labels) if labels else 0.0,
        "f1": mean(per_class[label]["f1"] for label in labels) if labels else 0.0,
    }

    return {
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro": macro,
        "accuracy": accuracy,
    }


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def compute_metrics(records: list[PredictionRecord]) -> dict:
    total = len(records)

    category_report = classification_report(
        list(CATEGORIES), [r.true_category for r in records], [r.pred_category for r in records]
    )
    urgency_report = classification_report(
        list(URGENCIES), [r.true_urgency for r in records], [r.pred_urgency for r in records]
    )
    needs_human_report = classification_report(
        ["true", "false"],
        ["true" if r.true_needs_human else "false" for r in records],
        ["true" if r.pred_needs_human else "false" for r in records],
    )

    rule_records = [r for r in records if r.source == "rule"]
    api_records = [r for r in records if r.source == "api"]

    rule_hit_rate = len(rule_records) / total if total else 0.0
    api_calls_made = len(api_records)
    api_calls_saved = len(rule_records)

    api_latencies = sorted(r.latency_ms for r in api_records if r.latency_ms is not None)
    p50 = _percentile(api_latencies, 0.50)
    p95 = _percentile(api_latencies, 0.95)

    schema_violations = sum(1 for r in api_records if not r.schema_valid)
    schema_violation_rate = schema_violations / len(api_records) if api_records else 0.0

    total_input_tokens = sum(r.input_tokens for r in api_records)
    total_output_tokens = sum(r.output_tokens for r in api_records)
    cost_usd = (
        total_input_tokens / 1_000_000 * settings.input_price_per_mtok_usd
        + total_output_tokens / 1_000_000 * settings.output_price_per_mtok_usd
    )
    cost_inr = cost_usd * settings.usd_to_inr
    cost_inr_per_1000_tickets = (cost_inr / total * 1000) if total else 0.0

    overall_accuracy = sum(1 for r in records if r.fully_correct) / total if total else 0.0

    return {
        "total_tickets": total,
        "overall_accuracy": overall_accuracy,
        "category": category_report,
        "urgency": urgency_report,
        "needs_human": needs_human_report,
        "rule_hit_rate": rule_hit_rate,
        "api_calls_made": api_calls_made,
        "api_calls_saved": api_calls_saved,
        "latency_ms": {"p50": p50, "p95": p95},
        "schema_violation_rate": schema_violation_rate,
        "schema_violations": schema_violations,
        "cost_inr_per_1000_tickets": cost_inr_per_1000_tickets,
    }


# --- diff / comparison between two runs -------------------------------------

# Metrics where a *lower* value is the improvement; everything else in
# DIFF_METRIC_PATHS is treated as higher-is-better.
_LOWER_IS_BETTER = {
    "schema_violation_rate",
    "cost_inr_per_1000_tickets",
    "latency_ms.p50",
    "latency_ms.p95",
    "guardrails.attack_success_rate",
}

DIFF_METRIC_PATHS: list[tuple[str, str]] = [
    ("Overall accuracy", "overall_accuracy"),
    ("Category macro F1", "category.macro.f1"),
    ("Urgency macro F1", "urgency.macro.f1"),
    ("Needs-human F1", "needs_human.macro.f1"),
    ("Rule hit rate", "rule_hit_rate"),
    ("API calls made", "api_calls_made"),
    ("Schema violation rate", "schema_violation_rate"),
    ("Cost / 1000 tickets (INR)", "cost_inr_per_1000_tickets"),
    ("Latency p50 (ms)", "latency_ms.p50"),
    ("Latency p95 (ms)", "latency_ms.p95"),
    ("Guardrail attack success rate", "guardrails.attack_success_rate"),
]


def _get_path(d: dict, path: str):
    node = d
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _is_correct(pred: dict) -> bool:
    return (
        pred["pred_category"] == pred["true_category"]
        and pred["pred_urgency"] == pred["true_urgency"]
        and pred["pred_needs_human"] == pred["true_needs_human"]
    )


def compute_diff(
    metrics_a: dict,
    metrics_b: dict,
    predictions_a: list[dict],
    predictions_b: list[dict],
) -> dict:
    """Metric-by-metric deltas plus the ticket ids that flipped correctness
    between two runs. Shared by the `diff` and `report` CLI commands.
    """
    rows = []
    for label, path in DIFF_METRIC_PATHS:
        a_val = _get_path(metrics_a, path)
        b_val = _get_path(metrics_b, path)
        delta = None if (a_val is None or b_val is None) else b_val - a_val
        rows.append(
            {
                "label": label,
                "path": path,
                "a": a_val,
                "b": b_val,
                "delta": delta,
                "higher_is_better": path not in _LOWER_IS_BETTER,
            }
        )

    preds_a = {p["id"]: p for p in predictions_a}
    preds_b = {p["id"]: p for p in predictions_b}
    common_ids = sorted(set(preds_a) & set(preds_b))

    flips_to_correct = []
    flips_to_incorrect = []
    for ticket_id in common_ids:
        correct_a = _is_correct(preds_a[ticket_id])
        correct_b = _is_correct(preds_b[ticket_id])
        if not correct_a and correct_b:
            flips_to_correct.append(ticket_id)
        elif correct_a and not correct_b:
            flips_to_incorrect.append(ticket_id)

    return {
        "rows": rows,
        "flips_to_correct": flips_to_correct,
        "flips_to_incorrect": flips_to_incorrect,
    }
