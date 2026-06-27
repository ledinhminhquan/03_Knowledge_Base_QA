.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-all data build-kb train-retriever train-reader \
        evaluate serve demo ask autopilot report slides grade test lint docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install (core deps) in editable mode
	pip install -r requirements.txt && pip install -e .

install-all: ## Install with all optional extras
	pip install -e ".[all,dev]"

data: ## Download datasets + build demo KB
	$(PY) -m kbqa.cli data --task all

build-kb: ## Build the demo knowledge-base index
	$(PY) -m kbqa.cli build-kb --config configs/train.yaml

train-retriever: ## Fine-tune the dense retriever
	$(PY) -m kbqa.cli train-retriever --config configs/train.yaml

train-reader: ## Fine-tune the extractive reader
	$(PY) -m kbqa.cli train-reader --config configs/train.yaml

evaluate: ## Retrieval + reader + e2e evaluation
	$(PY) -m kbqa.cli evaluate --config configs/train.yaml

serve: ## Start the FastAPI server (port 8000)
	$(PY) -m kbqa.cli serve --config configs/infer.yaml --host 0.0.0.0 --port 8000

demo: ## Launch the Gradio demo (port 7860)
	$(PY) app/gradio_app.py

ask: ## Demo the agent on built-in samples
	$(PY) -m kbqa.cli demo-agent --config configs/infer.yaml

autopilot: ## One button: build KB -> train -> eval -> analysis -> report + slides
	$(PY) -m kbqa.cli autopilot --config configs/train.yaml

report: ## Generate the PDF report
	$(PY) -m kbqa.cli generate-report --config configs/train.yaml

slides: ## Generate the PPTX slide deck
	$(PY) -m kbqa.cli generate-slides --config configs/train.yaml

grade: ## Rubric completeness self-check
	$(PY) -m kbqa.cli grade

test: ## Run the test suite
	pytest -q

lint: ## Lint with ruff
	ruff check src tests

docker: ## Build the Docker image
	docker build -t kbqa:1.0.0 .

clean: ## Remove caches + build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
