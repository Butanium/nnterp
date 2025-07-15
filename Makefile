.PHONY: dev
dev:
	@echo ">>> Installing development environment..."
	python -m pip install uv
	uv sync --all-extras
	uv pip install flash-attn --no-build-isolation
	@echo ">>> Done! Activate with 'source .venv/bin/activate'"

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + ; find . -name "*.pyc" -delete ; find . -name "*.pyo" -delete ; find . -type d -name ".pytest_cache" -exec rm -rf {} +