.PHONY: help install test lint format clean docs build

help:
	@echo "Available commands:"
	@echo "  make install    - Install the package and all dev dependencies"
	@echo "  make test       - Run all tests with coverage reporting"
	@echo "  make lint       - Run ruff linter and mypy type checker"
	@echo "  make format     - Format code using ruff"
	@echo "  make clean      - Remove build artifacts, pycache, and test outputs"
	@echo "  make docs       - Serve the MkDocs documentation site locally"
	@echo "  make build      - Build the PyPI distribution packages"

install:
	pip install -e ".[dev,all]"
	pre-commit install

test:
	pytest tests/ --cov=uchi --cov-report=term-missing

lint:
	ruff check uchi/
	mypy uchi/

format:
	ruff check uchi/ --fix

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	mkdocs serve

build: clean
	python -m build
