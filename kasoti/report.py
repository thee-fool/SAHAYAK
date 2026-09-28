"""Single self-contained HTML report: the same comparison as `kasoti diff`,
plus confusion matrices, generated as one file with no external assets."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

from kasoti.metrics import compute_diff


def _load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return metrics, predictions


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _delta_cell(delta: float | None, higher_is_better: bool) -> str:
    if delta is None:
        return '<td class="num">—</td>'
    if delta == 0:
        return '<td class="num">0.000</td>'
    improved = (delta > 0) == higher_is_better
    cls = "good" if improved else "bad"
    sign = "+" if delta > 0 else ""
    return f'<td class="num {cls}">{sign}{delta:.3f}</td>'


def _confusion_table(title: str, report: dict) -> str:
    labels = list(report["confusion_matrix"].keys())
    header = "".join(f'<th class="num">{html.escape(label)}</th>' for label in labels)
    body_rows = []
    for true_label in labels:
        cells = "".join(
            (
                f'<td class="num{" diag" if true_label == pred_label else ""}">'
                f'{report["confusion_matrix"][true_label][pred_label]}</td>'
            )
            for pred_label in labels
        )
        body_rows.append(f"<tr><th>{html.escape(true_label)}</th>{cells}</tr>")
    return (
        '<table class="confusion">'
        f"<caption>{html.escape(title)} — rows: true, columns: predicted</caption>"
        f'<thead><tr><th>true \\ pred</th>{header}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
    )


def generate_report(run_a: Path, run_b: Path, version_a: str, version_b: str, out: Path) -> None:
    metrics_a, preds_a = _load_run(run_a)
    metrics_b, preds_b = _load_run(run_b)
    diff = compute_diff(metrics_a, metrics_b, preds_a, preds_b)

    metric_rows = "".join(
        f"<tr><td>{html.escape(row['label'])}</td>"
        f'<td class="num">{_fmt(row["a"])}</td>'
        f'<td class="num">{_fmt(row["b"])}</td>'
        f"{_delta_cell(row['delta'], row['higher_is_better'])}</tr>"
        for row in diff["rows"]
    )

    flips_to_correct = ", ".join(diff["flips_to_correct"]) or "(none)"
    flips_to_incorrect = ", ".join(diff["flips_to_incorrect"]) or "(none)"

    guardrail_a = metrics_a.get("guardrails", {})
    guardrail_b = metrics_b.get("guardrails", {})
    attack_types = sorted(
        set(guardrail_a.get("attack_success_by_type", {})) | set(guardrail_b.get("attack_success_by_type", {}))
    )
    guardrail_rows = "".join(
        f"<tr><td>{html.escape(attack_type)}</td>"
        f'<td class="num">{_fmt(guardrail_a.get("attack_success_by_type", {}).get(attack_type))}</td>'
        f'<td class="num">{_fmt(guardrail_b.get("attack_success_by_type", {}).get(attack_type))}</td></tr>'
        for attack_type in attack_types
    )

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    va, vb = html.escape(version_a), html.escape(version_b)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kasoti report: {va} vs {vb}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 960px;
          margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #8884; padding-bottom: .25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .75rem 0; }}
  th, td {{ border: 1px solid #8886; padding: .35rem .6rem; text-align: left; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.good {{ color: #1a7f37; font-weight: 600; }}
  td.bad {{ color: #cf222e; font-weight: 600; }}
  td.diag {{ background: #80808022; font-weight: 600; }}
  .meta {{ color: #808080; font-size: .9rem; }}
  .flips {{ font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: .85rem; word-break: break-word; }}
  caption {{ caption-side: top; text-align: left; font-weight: 600; margin-bottom: .25rem; }}
  table.confusion {{ display: inline-block; margin-right: 1.5rem; vertical-align: top; max-width: 100%; overflow-x: auto; }}
</style>
</head>
<body>
<h1>kasoti regression report: {va} vs {vb}</h1>
<p class="meta">Generated {generated_at} &middot; run A: {html.escape(run_a.name)} &middot; run B: {html.escape(run_b.name)}</p>

<h2>Metrics</h2>
<table>
  <thead><tr><th>Metric</th><th class="num">{va}</th><th class="num">{vb}</th><th class="num">delta</th></tr></thead>
  <tbody>{metric_rows}</tbody>
</table>

<h2>Confusion matrices &mdash; category</h2>
<div>
{_confusion_table(version_a, metrics_a["category"])}
{_confusion_table(version_b, metrics_b["category"])}
</div>

<h2>Confusion matrices &mdash; urgency</h2>
<div>
{_confusion_table(version_a, metrics_a["urgency"])}
{_confusion_table(version_b, metrics_b["urgency"])}
</div>

<h2>Guardrail: attack success rate by type</h2>
<table>
  <thead><tr><th>Attack type</th><th class="num">{va}</th><th class="num">{vb}</th></tr></thead>
  <tbody>{guardrail_rows}</tbody>
</table>

<h2>Flipped tickets</h2>
<p><strong>Incorrect &rarr; correct</strong> ({len(diff["flips_to_correct"])}):</p>
<p class="flips">{html.escape(flips_to_correct)}</p>
<p><strong>Correct &rarr; incorrect</strong> ({len(diff["flips_to_incorrect"])}):</p>
<p class="flips">{html.escape(flips_to_incorrect)}</p>

</body>
</html>
"""
    out.write_text(html_doc, encoding="utf-8")
