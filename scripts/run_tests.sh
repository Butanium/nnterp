#!/bin/bash
make clean
uv run pytest nnterp/tests/ --cache-clear "$@"