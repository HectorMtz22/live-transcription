default:
    @just --list

sync:
    uv sync --all-packages

run *ARGS:
    uv run --package live-transcribe-cli live-transcribe {{ARGS}}

fmt:
    uv run ruff format packages/

lint:
    uv run ruff check packages/

clean:
    rm -rf .venv packages/core/build packages/cli/build
    find packages -name '__pycache__' -type d -exec rm -rf {} +
