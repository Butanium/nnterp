# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `uv install` - Install dependencies
- `uv run python -m pytest` - Run all tests
- `uv run python -m pytest tests/test_interventions.py` - Run specific test file
- `uv run python -m pytest tests/test_interventions.py::test_logit_lens` - Run specific test

### Code Quality
- `uv run black .` - Format code with Black (line length 88)
- `uv run python -m build` - Build package for distribution

### Documentation
- `cd docs && make html` - Build Sphinx documentation
- `cd docs && make clean` - Clean documentation build files

## Architecture Overview

nnterp is a mechanistic interpretability library built on top of nnsight, providing a unified interface for transformer analysis through several key components:

### Core Components

**StandardizedTransformer** (`standardized_transformer.py`)
- Unified interface for different transformer architectures
- Standardizes module naming across models (layers, attention, MLP components)
- Provides convenient accessors: `model.layers_input[i]`, `model.attentions_output[i]`, `model.mlps_output[i]`
- Includes built-in steering methods via `model.steer()`
- **Primary model loading method**: Use `StandardizedTransformer("model_name")` instead of deprecated `load_model()`

**Intervention Framework** (`interventions.py`)
- **Logit Lens**: Projects hidden states to vocabulary at each layer
- **Patchscope**: Compares activations between different contexts using `TargetPrompt`
- **Activation Steering**: Direct behavior modification via `steer()` function
- **Latent Prompts**: Advanced prompt manipulation with `LatentPrompt` and `run_latent_prompt()`

**Utilities** (`nnsight_utils.py`)
- Core activation collection: `get_token_activations()` and `collect_activations_batched()`
- Utility functions: `get_num_layers()`, `project_on_vocab()`, `get_next_token_probs()`

**Prompt Management** (`prompt_utils.py`)
- `Prompt` class for tracking target tokens
- `run_prompts()` for batch evaluation with target tracking

### Key Design Patterns

1. **Batched Processing**: Most functions support both single inputs and batched operations
2. **Device Management**: Automatic GPU/CPU handling with proper tensor movement
3. **Standardized Interfaces**: Consistent API across different model architectures
4. **Context Management**: Heavy use of `model.trace()` context for interventions

### Test Structure

Tests are organized around intervention types:
- `test_interventions.py`: Core intervention methods (logit lens, patchscope, steering)  
- `test_model_renaming.py`: Module standardization functionality
- `test_nnsight_utils.py`: Core utility functions

Test fixtures use multiple model architectures: `["Maykeye/TinyLLama-v0", "gpt2", "bigscience/bigscience-small-testing"]`

### Dependencies

- **Core**: `nnsight>=0.5.0dev2` (the main dependency for model tracing)
- **Visualization**: `plotly`, `pandas` (install with `pip install nnterp[display]`)
- **Development**: `pytest`, `black`, `sphinx` (install with `pip install nnterp[dev]`)

### Common Patterns

**Model Loading and Usage**:
```python
from nnterp import StandardizedTransformer
model = StandardizedTransformer("gpt2")
```

**Intervention Structure**:
```python
with model.trace(prompts) as tracer:
    # Access activations via standardized names
    activations = model.layers_output[layer_idx].save()
    # Apply interventions
    steer(model, layers=layer_idx, steering_vector=vector)
```

**Target Token Tracking**:
```python
from nnterp import Prompt, run_prompts
prompts = [Prompt.from_strings("input", {"target": "expected"}, tokenizer)]
results = run_prompts(model, prompts)
```