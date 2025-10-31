.PHONY: help install test lint format clean bench all

help:
	@echo "Multi-Bin Batching Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make bench        - Run benchmarks"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make all          - Format, lint, and test"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest mbb_core/tests/ -v --cov=mbb_core --cov-report=term-missing

test-quick:
	pytest mbb_core/tests/ -v

lint:
	ruff check mbb_core/ bench/ adapters/
	black --check mbb_core/ bench/ adapters/

format:
	ruff check --fix mbb_core/ bench/ adapters/
	black mbb_core/ bench/ adapters/

bench:
	python bench/loadgen.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -delete

all: format lint test

.DEFAULT_GOAL := help

