.PHONY install test lint format check notebook

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check .
	uv run pytest

notebook:
	uv run jupyter lab