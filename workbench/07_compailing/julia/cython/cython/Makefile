.PHONY: lint fmt help test lock setup train build build6
.DEFAULT_GOAL := help

lint: ## Run Linter
	uv run ruff check .
	uv run mypy .

fmt: ## Run formatter
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run tests
	uv run pytest .

build: ## Train the model
	uv run setup.py build_ext --inplace

build6: ## Train the model
	uv run setup6.py build_ext --inplace

help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
