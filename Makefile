.PHONY: help install lint format test check offline quick quick-retrieval chunking embeddings clean

PY ?= python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install the package with dev extras (use inside a virtualenv)
	$(PY) -m pip install -e ".[dev]"

lint:  ## Lint and check formatting
	ruff check .
	ruff format --check .

format:  ## Auto-format and fix lint issues
	ruff format .
	ruff check --fix .

test:  ## Run unit tests (no Ollama needed)
	$(PY) -m pytest

check:  ## Verify Ollama is running and the models for configs/quick.yaml are pulled
	rag-eval check configs/quick.yaml

offline:  ## End-to-end smoke run with no models (hashing embedder + heuristic LLM)
	rag-eval run configs/offline.yaml

quick-retrieval:  ## Quick comparison, retrieval metrics only (needs Ollama + nomic-embed-text)
	rag-eval run configs/quick.yaml --retrieval-only

quick:  ## Quick comparison incl. generation and LLM judge (needs Ollama models)
	rag-eval run configs/quick.yaml

chunking:  ## Chunking strategy/size sweep, retrieval only
	rag-eval run configs/chunking-sweep.yaml --retrieval-only

embeddings:  ## Embedding model sweep (retrieval only)
	rag-eval run configs/embedding-sweep.yaml

clean:  ## Remove caches and results
	rm -rf .cache .pytest_cache .ruff_cache results/*/
