# ==========================================
# RCPE - DEVELOPMENT MAKEFILE
# ==========================================

.PHONY: setup test lint clean deploy help

PYTHON = python3
PIP = pip
PYTEST = pytest
MYPY = mypy
BLACK = black
ISORT = isort

help:
	@echo "RCPE Development Commands:"
	@echo "  setup    Install dependencies and set up the environment"
	@echo "  test     Run the full test suite"
	@echo "  lint     Run linters (black, isort, mypy)"
	@echo "  clean    Remove temporary files and caches"
	@echo "  deploy   Run the application in production mode"

setup:
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi

test:
	$(PYTEST) tests/

lint:
	$(BLACK) .
	$(ISORT) .
	$(MYPY) src/rcpe/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

deploy:
	docker-compose up --build -d
