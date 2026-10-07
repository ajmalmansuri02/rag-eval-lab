"""Command-line interface: ``rag-eval run|chunks|check``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from rag_eval_lab import __version__
from rag_eval_lab.chunking import ChunkingConfig, chunk_document
from rag_eval_lab.config import PROVIDERS, load_config
from rag_eval_lab.corpus import load_corpus
from rag_eval_lab.env import env, load_dotenv
from rag_eval_lab.http import ProviderError
from rag_eval_lab.report import markdown_table, write_report
from rag_eval_lab.runner import Runner


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, provider=args.provider)
    if args.limit is not None:
        config.limit = args.limit
    if args.retrieval_only:
        config.generate = False
        config.judge = False
    if args.no_judge:
        config.judge = False
    if args.experiment:
        wanted = set(args.experiment)
        config.experiments = [e for e in config.experiments if e.name in wanted]
        if not config.experiments:
            print(f"No experiments match {sorted(wanted)}", file=sys.stderr)
            return 2
    result = Runner(config).run()
    out = write_report(result, Path(args.output_dir) if args.output_dir else None)
    print(markdown_table(result.summaries))
    print(f"\nWrote results to {out}")
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    """Print how a document is chunked, to build intuition for chunking settings."""
    docs = {d.doc_id: d for d in load_corpus(args.corpus_dir)}
    if args.doc not in docs:
        print(f"Unknown doc {args.doc!r}. Available: {', '.join(sorted(docs))}", file=sys.stderr)
        return 2
    cfg = ChunkingConfig(strategy=args.strategy, chunk_size=args.size, overlap=args.overlap)
    chunks = chunk_document(docs[args.doc], cfg)
    for chunk in chunks:
        words = len(chunk.text.split())
        print(f"--- {chunk.chunk_id} ({words} words) ---")
        print(chunk.text)
        print()
    print(f"{len(chunks)} chunk(s) with {cfg.label}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Check that Ollama is reachable and the models referenced by a config are pulled."""
    config = load_config(args.config)
    base = (env("OLLAMA_BASE_URL", "http://localhost:11434") or "").rstrip("/")
    needed = {
        e.embedding.model
        for e in config.experiments
        if e.uses_embeddings and e.embedding.provider == "ollama"
    }
    for llm_cfg, used in ((config.generator, config.generate), (config.judge_llm, config.judge)):
        if used and llm_cfg.provider == "ollama":
            needed.add(llm_cfg.model)
    if any(e.rerank.enabled for e in config.experiments) and config.generator.provider == "ollama":
        needed.add(config.generator.model)
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Cannot reach Ollama at {base}: {exc}")
        print("Start it with `ollama serve`, or run with --provider offline.")
        return 1
    have = {m["name"] for m in tags.get("models", [])}
    have |= {name.removesuffix(":latest") for name in have}
    missing = sorted(m for m in needed if m not in have)
    print(f"Ollama reachable at {base}. Models needed: {', '.join(sorted(needed)) or 'none'}")
    if missing:
        print("Missing models, pull them with:")
        for model in missing:
            print(f"  ollama pull {model}")
        return 1
    print("All models available.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the experiments in a config file")
    run.add_argument("config", help="path to a YAML config, e.g. configs/quick.yaml")
    run.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="override every model provider (offline = no network, smoke test)",
    )
    run.add_argument("--limit", type=int, help="only use the first N golden questions")
    run.add_argument(
        "--experiment", action="append", help="run only the named experiment (repeatable)"
    )
    run.add_argument(
        "--retrieval-only", action="store_true", help="skip generation and judging (fast)"
    )
    run.add_argument("--no-judge", action="store_true", help="generate answers but skip judging")
    run.add_argument("--output-dir", help="override the config's output_dir")
    run.set_defaults(func=cmd_run)

    chunks = sub.add_parser("chunks", help="show how one document gets chunked")
    chunks.add_argument("doc", help="document id, e.g. pricing-plans")
    chunks.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    chunks.add_argument("--size", type=int, default=200)
    chunks.add_argument("--overlap", type=int, default=20)
    chunks.add_argument("--corpus-dir", default="data/corpus")
    chunks.set_defaults(func=cmd_chunks)

    check = sub.add_parser("check", help="verify Ollama is running and models are pulled")
    check.add_argument("config", nargs="?", default="configs/quick.yaml")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        print("Is Ollama running? Try `rag-eval check`, or `--provider offline`.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
