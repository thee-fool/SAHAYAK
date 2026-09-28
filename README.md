# kasoti

A prompt regression testing and guardrail harness for LLM classification tasks.
The demo task is customer support ticket triage: classify a ticket into a
`category`, `urgency`, and whether it `needs_human` review — then compare two
versions of the classification prompt on accuracy, cost, latency, and
resistance to prompt injection.

```
kasoti generate-data
kasoti run --prompt kasoti/prompts/triage_v1.yaml --mock
kasoti run --prompt kasoti/prompts/triage_v2.yaml --mock
kasoti diff v1 v2
kasoti report v1 v2 --out report.html
```

## Architecture

```
ticket text
    |
    v
 rules.py -------- regex/keyword pre-filter, resolves "obvious" tickets
    | (no match)     for free, with zero risk of prompt injection since
    v                 it never calls a model
 runner.py -------- on-disk JSON cache (sha256 of prompt config + text)
    | (cache miss)
    v
 Anthropic API ---- asyncio, semaphore(8), exponential backoff on 429/5xx
    |
    v
 pydantic validation (ClassificationOutput) -- violations counted, not raised
    |
    v
 predictions.jsonl + raw_responses.jsonl + resolved_prompt.yaml + metrics.json
    |                                                 (runs/<version>_<ts>/)
    v
 metrics.py ------- precision/recall/F1, confusion matrix, cost, latency
 guardrails.py ---- runs data/injections.jsonl through the same pipeline,
                     scores attack success rate (predicted != true label)
    |
    v
 kasoti diff / kasoti report -- metric-by-metric comparison between two
                                 prompt versions, plus which ticket ids
                                 flipped correct <-> incorrect
```

**Module map**

| Module | Responsibility |
|---|---|
| `models.py` | Shared pydantic models: `Ticket`, `ClassificationOutput` (the canonical response schema), `PromptConfig`, `PredictionRecord`. |
| `config.py` | `pydantic-settings`: paths, concurrency cap, retry/backoff, pricing constants. |
| `datagen.py` | Deterministic, templated generator for `data/golden.jsonl` (150 tickets, ~25 ambiguous) and `data/injections.jsonl` (15 tickets). No LLM calls — same seed always produces the same data. |
| `rules.py` | Regex/keyword pre-filter. Precision-first: each rule matches one specific unambiguous phrasing, and an explicit hedge list defers anything that mixes signals to the model. |
| `runner.py` | Async execution: rules-first dispatch, on-disk cache, retries with backoff, schema validation, `--mock` mode, run persistence. |
| `metrics.py` | Pure functions: confusion matrix + precision/recall/F1, rule hit rate, cost, latency percentiles, schema violation rate, and the `diff` computation shared by the `diff`/`report` commands. |
| `guardrails.py` | A regex injection *detector* (independent of the model) plus `run_guardrail_eval`, which scores attack success rate on `injections.jsonl`. |
| `cli.py` | Typer app wiring the above into `generate-data` / `run` / `diff` / `report`. |
| `report.py` | Renders the diff as one self-contained HTML file (inline CSS, no external assets). |

**Why rules run first:** a keyword match never calls an LLM, so it can't be
talked out of its answer by an injected instruction. It's also free and
instant. `rules.py` is deliberately conservative — narrow patterns, one
scenario per rule, and a hedge list that defers anything ambiguous — so the
~25 deliberately-ambiguous golden tickets and all of the crafted injection
tickets fall through to the model, where the real accuracy/robustness
comparison between prompt versions actually happens.

**Prompt versions:** `triage_v1.yaml` is a naive, minimal system prompt with
no few-shot examples. `triage_v2.yaml` adds explicit category/urgency
constraints, five few-shot examples (including one injection example), and an
instruction that the ticket text — wrapped in `<<<TICKET_START>>>` /
`<<<TICKET_END>>>` delimiters in the user message — is untrusted data, never
an instruction. The ticket text is never interpolated into the system prompt.

## Quickstart

```bash
uv sync --dev
uv run kasoti generate-data
uv run kasoti run --prompt kasoti/prompts/triage_v1.yaml --mock
uv run kasoti run --prompt kasoti/prompts/triage_v2.yaml --mock
uv run kasoti diff v1 v2
uv run kasoti report v1 v2 --out report.html
uv run pytest -q
uv run ruff check .
```

`--mock` returns deterministic fake classifications (seeded per ticket id) so
the entire pipeline — rules, cache, metrics, guardrails, diff, report — runs
end to end with **zero API calls and zero cost**. The mock model is modeled
as noisier and more prompt-injection-susceptible for v1 than v2, so the demo
reproduces the kind of improvement a real "add constraints + untrusted-data
framing" prompt revision should achieve.

To run against the real API instead, set `ANTHROPIC_API_KEY` and drop
`--mock`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run kasoti run --prompt kasoti/prompts/triage_v1.yaml
```

Every ticket is cached on disk under `.cache/` keyed by
`sha256(prompt_config + ticket_text)`, so re-running the same prompt version
against the same data costs nothing after the first pass.

## Runs

Each `kasoti run` writes a self-contained, reproducible directory:

```
runs/<version>_<timestamp>/
  predictions.jsonl     # one row per ticket: true/predicted labels, source, latency, tokens
  raw_responses.jsonl   # raw model/rule output per ticket
  resolved_prompt.yaml  # the exact prompt config used
  metrics.json          # everything metrics.py + guardrails.py computed, including guardrails
```

`kasoti diff`/`kasoti report` always compare the *latest* run directory for
each version prefix (e.g. `v1_20240521T101500Z`).

## Limitations / what I'd add next

- **Synthetic data isn't real data.** `datagen.py` is templated and seeded for
  reproducibility, not sampled from an actual support queue — label
  distribution and phrasing variety won't match production traffic. The
  intended reproducibility/offline-testing property still holds; the
  intended realism is only approximate.
- **The mock model's error/injection-susceptibility rates are hand-picked**
  to make the v1→v2 story clear, not measured. They're a stand-in for what a
  real run against the Anthropic API would show, not a claim about it — the
  same commands work unmodified against the real API once you have a key.
- **The regex guardrail detector is a heuristic**, not a defense — it flags
  known injection phrasings but a sufficiently novel or obfuscated payload
  would slip past it undetected (independent of whether the *classification*
  itself resists it). The `rules.py` pre-filter is the only layer that's
  structurally immune to injection, because it never calls a model.
- **Cost estimate uses fixed, hardcoded USD/token pricing and a fixed
  USD→INR rate** (`config.py`), both of which drift in reality; treat
  `cost_inr_per_1000_tickets` as directionally useful, not exact.
- **No persistent datastore** — everything is JSONL/YAML/JSON files under
  `runs/`, `.cache/`, and `data/`. Fine at this scale, wouldn't scale to a
  shared multi-user setup.
- **What I'd add next:** a real small labeled sample (even 30-50 tickets) run
  against the live API to validate the mock model's assumptions; a
  configurable rate-limit-aware batch scheduler instead of a flat semaphore;
  golden-set drift detection (alert when `diff` crosses a threshold, for CI);
  and a proper LLM-based judge for the ambiguous tickets where "correct"
  label is itself debatable, rather than a single fixed ground truth.
