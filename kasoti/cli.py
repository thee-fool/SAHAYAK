"""kasoti CLI: generate-data, run, diff, report."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kasoti.config import settings

app = typer.Typer(add_completion=False, help="kasoti: prompt regression testing & guardrail harness")
console = Console()


def _find_latest_run(version: str) -> Path:
    candidates = sorted(settings.runs_dir.glob(f"{version}_*"))
    if not candidates:
        raise typer.BadParameter(
            f"No runs found for '{version}' under {settings.runs_dir}/. "
            f"Run `kasoti run --prompt kasoti/prompts/triage_{version}.yaml --mock` first."
        )
    return candidates[-1]


def _load_run_data(run_dir: Path) -> tuple[dict, list[dict]]:
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
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_delta(delta: float | None, higher_is_better: bool) -> str:
    if delta is None:
        return "—"
    if delta == 0:
        return "0.000"
    improved = (delta > 0) == higher_is_better
    sign = "+" if delta > 0 else ""
    text = f"{sign}{delta:.3f}"
    color = "green" if improved else "red"
    return f"[{color}]{text}[/{color}]"


@app.command("generate-data")
def generate_data_cmd(
    seed: int = typer.Option(settings.data_seed, "--seed", help="Seed for reproducible data generation"),
) -> None:
    """Generate data/golden.jsonl and data/injections.jsonl."""
    from kasoti.datagen import generate_all

    n_golden, n_injections = generate_all(settings.data_dir, seed=seed)
    console.print(f"[green]Generated[/green] {n_golden} golden tickets -> {settings.data_dir / 'golden.jsonl'}")
    console.print(f"[green]Generated[/green] {n_injections} injection tickets -> {settings.data_dir / 'injections.jsonl'}")


@app.command("run")
def run_cmd(
    prompt: Path = typer.Option(..., "--prompt", exists=True, help="Path to a prompts/*.yaml config"),
    mock: bool = typer.Option(False, "--mock", help="Deterministic fake classifications, zero API calls"),
) -> None:
    """Run the full pipeline for one prompt version against the golden set."""
    from kasoti.runner import execute_run

    run_dir = execute_run(prompt, mock=mock)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    console.print(f"[green]Run complete[/green] -> {run_dir}")
    console.print(f"  overall accuracy:      {metrics['overall_accuracy']:.1%}")
    console.print(
        f"  rule hit rate:         {metrics['rule_hit_rate']:.1%}  "
        f"(api calls made: {metrics['api_calls_made']}, saved: {metrics['api_calls_saved']})"
    )
    console.print(f"  schema violation rate: {metrics['schema_violation_rate']:.1%}")
    console.print(f"  cost / 1000 tickets:   ₹{metrics['cost_inr_per_1000_tickets']:.2f}")
    console.print(f"  guardrail attack success rate: {metrics['guardrails']['attack_success_rate']:.1%}")


@app.command("diff")
def diff_cmd(
    version_a: str = typer.Argument(..., metavar="V1"),
    version_b: str = typer.Argument(..., metavar="V2"),
) -> None:
    """Compare the latest runs of two prompt versions."""
    from kasoti.metrics import compute_diff

    run_a = _find_latest_run(version_a)
    run_b = _find_latest_run(version_b)
    metrics_a, preds_a = _load_run_data(run_a)
    metrics_b, preds_b = _load_run_data(run_b)
    diff = compute_diff(metrics_a, metrics_b, preds_a, preds_b)

    console.print(f"Comparing [bold]{version_a}[/bold] ({run_a.name}) -> [bold]{version_b}[/bold] ({run_b.name})\n")

    table = Table(title=f"{version_a} vs {version_b}")
    table.add_column("Metric")
    table.add_column(version_a, justify="right")
    table.add_column(version_b, justify="right")
    table.add_column("delta", justify="right")

    for row in diff["rows"]:
        table.add_row(
            row["label"],
            _fmt(row["a"]),
            _fmt(row["b"]),
            _fmt_delta(row["delta"], row["higher_is_better"]),
        )

    console.print(table)

    to_correct = diff["flips_to_correct"]
    to_incorrect = diff["flips_to_incorrect"]
    console.print(
        f"\n[green]Flipped incorrect -> correct[/green] ({len(to_correct)}): "
        f"{', '.join(to_correct) if to_correct else '(none)'}"
    )
    console.print(
        f"[red]Flipped correct -> incorrect[/red] ({len(to_incorrect)}): "
        f"{', '.join(to_incorrect) if to_incorrect else '(none)'}"
    )


@app.command("report")
def report_cmd(
    version_a: str = typer.Argument(..., metavar="V1"),
    version_b: str = typer.Argument(..., metavar="V2"),
    out: Path = typer.Option(Path("report.html"), "--out", help="Output HTML file path"),
) -> None:
    """Generate a single self-contained HTML report comparing two prompt versions."""
    from kasoti.report import generate_report

    run_a = _find_latest_run(version_a)
    run_b = _find_latest_run(version_b)
    generate_report(run_a, run_b, version_a, version_b, out)
    console.print(f"[green]Report written[/green] -> {out}")


if __name__ == "__main__":
    app()
