make clean
uv run pytest nnterp/tests/ --cache-clear --cov=nnterp --cov-report=html --cov-report=term --cov-config=pyproject.toml "$@"