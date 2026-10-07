from pathlib import Path

from rag_eval_lab.cli import main
from tests.conftest import CORPUS_DIR, REPO_ROOT


def test_cli_run_offline(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    code = main(
        [
            "run",
            "configs/quick.yaml",
            "--provider",
            "offline",
            "--limit",
            "5",
            "--experiment",
            "bm25",
            "--experiment",
            "hybrid-rrf",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| bm25 |" in out and "| hybrid-rrf |" in out
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1 and (run_dirs[0] / "summary.csv").is_file()


def test_cli_unknown_experiment(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert main(["run", "configs/quick.yaml", "--provider", "offline", "--experiment", "x"]) == 2


def test_cli_missing_config_is_a_clean_error(capsys):
    assert main(["run", "does-not-exist.yaml"]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_chunks(capsys):
    code = main(["chunks", "pricing-plans", "--size", "60", "--corpus-dir", str(CORPUS_DIR)])
    assert code == 0
    out = capsys.readouterr().out
    assert "pricing-plans#0" in out and "chunk(s) with recursive-60/20" in out
