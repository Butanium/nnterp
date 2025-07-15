#!/bin/bash
uv run pytest nnterp/tests/ --cache-clear "$@"