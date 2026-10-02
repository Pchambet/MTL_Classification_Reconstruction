.PHONY: setup data run report test lint all

setup:  ## install the locked environment
	uv sync --locked

data:  ## decode the committed images once into data/interim/
	uv run mtl-eurosat data

run:  ## multi-seed grid -> results/ (about 4 h on an Apple M-series GPU, resumable)
	uv run mtl-eurosat run

report:  ## figures -> docs/figures/, HTML report -> site/index.html
	uv run mtl-eurosat report

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

all: setup data run report
