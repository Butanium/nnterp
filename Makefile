.PHONY: dev
dev:
	@echo ">>> Installing development environment..."
	python -m pip install uv
	uv sync --all-extras
	uv pip install flash-attn --no-build-isolation
	@echo ">>> Done! Activate with 'source .venv/bin/activate'"

.PHONY: clean
clean:
	rm -rf .venv