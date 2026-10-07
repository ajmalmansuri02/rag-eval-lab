"""Writing results: markdown table, CSV files, a JSON snapshot and a chart."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.runner import ExperimentSummary, QuestionResult, RunResult

# (attribute, header, higher_is_better) for the markdown table
_COLUMNS: list[tuple[str, str, bool | None]] = [
    ("experiment", "Experiment", None),
    ("chunking", "Chunking", None),
    ("embedding", "Embedding", None),
    ("retrieval", "Retrieval", None),
    ("rerank", "Rerank", None),
    ("num_chunks", "Chunks", None),
    ("recall_at_k", "Recall@k", True),
    ("mrr", "MRR", True),
    ("hit_rate", "Hit@k", True),
    ("faithfulness", "Faithful", True),
    ("correctness", "Correct", True),
    ("retrieval_ms_mean", "Retr ms", False),
    ("retrieval_ms_p95", "Retr p95", False),
    ("generation_ms_mean", "Gen ms", False),
]


def _fmt(value: object, attr: str) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if "_ms_" in attr:
            return f"{value:.0f}" if value >= 100 else f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def markdown_table(summaries: list[ExperimentSummary]) -> str:
    """Render summaries as a markdown table; the best value of each metric column is bolded."""
    best: dict[str, float] = {}
    for attr, _, higher in _COLUMNS:
        if higher is None:
            continue
        values = [getattr(s, attr) for s in summaries if getattr(s, attr) is not None]
        if values:
            best[attr] = max(values) if higher else min(values)
    header = "| " + " | ".join(h for _, h, _ in _COLUMNS) + " |"
    divider = "|" + "|".join("---:" if h is not None else "---" for _, _, h in _COLUMNS) + "|"
    lines = [header, divider]
    for s in summaries:
        cells = []
        for attr, _, _ in _COLUMNS:
            value = getattr(s, attr)
            cell = _fmt(value, attr)
            if attr in best and value is not None and _fmt(value, attr) == _fmt(best[attr], attr):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_markdown(result: RunResult) -> str:
    cfg = result.config
    n_questions = result.summaries[0].num_questions if result.summaries else 0
    gen = f"{cfg.generator.provider}:{cfg.generator.label}" if cfg.generate else "disabled"
    judge = f"{cfg.judge_llm.provider}:{cfg.judge_llm.label}" if cfg.judge else "disabled"
    parts = [
        f"# Results: {cfg.name}",
        "",
        f"- Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Questions: {n_questions} (retrieval metrics use answerable questions only)",
        f"- top_k: {cfg.top_k}",
        f"- Generator: {gen}",
        f"- Judge: {judge}",
        "",
        markdown_table(result.summaries),
        "",
        "Retrieval metrics are document-level. Faithfulness and correctness are LLM-judge scores "
        "on a 1-5 rubric normalised to 0-1. Latencies are per question in milliseconds. "
        "**Bold** marks the best value in each column.",
        "",
    ]
    if cfg.generator.provider == "offline":
        parts += [
            "> Offline mode: hashing embeddings and a heuristic stand-in for the LLM. "
            "Answer-quality scores here are NOT meaningful; use this run only as a smoke test.",
            "",
        ]
    return "\n".join(parts)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rounded(row: dict[str, object]) -> dict[str, object]:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def _question_row(q: QuestionResult) -> dict[str, object]:
    row = asdict(q)
    for key in ("tags", "expected_doc_ids", "retrieved_doc_ids", "retrieved_chunk_ids"):
        row[key] = ";".join(row[key])
    return _rounded(row)


def write_chart(summaries: list[ExperimentSummary], path: Path) -> bool:
    """Draw a three-panel horizontal bar chart. Returns False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    names = [s.experiment for s in summaries][::-1]
    panels = [
        (
            "Retrieval (higher is better)",
            [
                ("recall_at_k", "Recall@k", "#2a78d6"),
                ("mrr", "MRR", "#eb6834"),
                ("hit_rate", "Hit@k", "#1baf7a"),
            ],
            (0, 1.0),
        ),
        (
            "Answer quality, LLM judge (higher is better)",
            [
                ("correctness", "Correctness", "#4a3aa7"),
                ("faithfulness", "Faithfulness", "#e87ba4"),
            ],
            (0, 1.0),
        ),
        (
            "Mean latency per question, ms (lower is better)",
            [
                ("retrieval_ms_mean", "Retrieval", "#2a78d6"),
                ("generation_ms_mean", "Generation", "#eb6834"),
            ],
            None,
        ),
    ]
    height = max(3.5, 0.45 * len(names) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, height), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, metrics, xlim) in zip(axes, panels, strict=True):
        present = [m for m in metrics if any(getattr(s, m[0]) is not None for s in summaries)]
        ax.set_facecolor("#fcfcfb")
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        if not present:
            ax.text(
                0.5,
                0.5,
                "not measured",
                ha="center",
                va="center",
                color=muted,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
        bar_h = 0.8 / max(len(present), 1)
        for j, (attr, label, color) in enumerate(present):
            values = [getattr(s, attr) or 0.0 for s in summaries][::-1]
            # First metric on top within each group, matching the legend order.
            ys = [i + ((len(present) - 1) / 2 - j) * bar_h for i in range(len(names))]
            ax.barh(ys, values, height=bar_h * 0.85, color=color, label=label)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9, color=ink)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        if present:
            ax.legend(
                fontsize=8,
                frameon=False,
                loc="upper left",
                ncol=len(present),
                bbox_to_anchor=(0, -0.08 * 6 / height),
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_report(result: RunResult, output_root: Path | None = None) -> Path:
    """Write all outputs into ``<output_dir>/<run-name>-<timestamp>/`` and return that path."""
    root = output_root or result.config.output_dir
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = Path(root) / f"{result.config.name}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    (out / "summary.md").write_text(summary_markdown(result), encoding="utf-8")
    summary_fields = [f.name for f in fields(ExperimentSummary)]
    write_csv(out / "summary.csv", [_rounded(asdict(s)) for s in result.summaries], summary_fields)
    question_fields = [f.name for f in fields(QuestionResult)]
    write_csv(
        out / "per_question.csv", [_question_row(q) for q in result.questions], question_fields
    )
    (out / "run.json").write_text(json.dumps(result.config.to_dict(), indent=2), encoding="utf-8")
    write_chart(result.summaries, out / "chart.png")
    return out
