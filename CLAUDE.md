# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Guidelines
- When asked a yes/no question, think carefully before responding. Do not start with yes/no
- Avoid sycophancy: if I challenge a claim you made, or suggest something, you should not assume I'm right.
- If (and only if) you feel like a question is hard or ambiguous, feel free to first propose a plan and wait for my feedback before implementing it.
- Always explain your reasoning and design choices

## Communication Style
- **Focus on assumptions, not summaries**: When completing code changes, highlight the key assumptions you made during implementation rather than listing what files were edited
- **Assumption-driven responses**: Structure responses around design decisions and assumptions rather than mechanical descriptions of changes
- **Example**: Instead of "I edited file X to add function Y", say "Key assumption: StandardizedTransformer failures should not update JSON since they're test-time convenience checks, not core loading capabilities"

## Sphinx Documentation Guidelines
- **IMPORTANT: Respect nnsight execution order**: ALL code examples must access components in forward pass order (layers_output[1] before layers_output[2], attention before layer output of same layer, etc.)
- **Use demo.py tone**: Keep explanations factual and concise, avoid verbose language like "core philosophy" or "research-first design"
- **nnterp is for transformers**: Describe it as a nnsight wrapper specifically for transformer models, not general mechanistic interpretability

## Code Philosophy
- Correctness first: Ensure code is functionally correct before optimizing
- Iterative refinement: After implementing changes, review the entire file to identify opportunities for simplification and improvement
- Use type hints and docstrings to enhance code clarity

## Research Context
You assist me - a researcher - with a research oriented library, not production systems. This context allows for specific approaches:
- Make reasonable assumptions based on common research practices and my instructions. Avoid writting fallbacks in case something is missing. THIS IS VERY IMPORTANT as you shouldn't create bloated code!
- Fail fast philosophy: Design code to crash immediately when assumptions are violated rather than silently handling errors. This means that you should only use try/catch blocks if it explicitely benefits the code logic. No need to state this in comments. DON'T WRITE FALLBACKS FOR NON-COMMON INPUTS! Instead write asserts for you assumptions. This is very important!
        - Example: Let the code fail if apply_chat_template doesn't exist rather than adding try-catch blocks
- Assumption hierarchy:
       - Minor assumptions: State them in your responses (not in code) and proceed
       - Major assumptions: Ask for confirmation before proceeding. Depending on the severity state them in code using comments.
- If you are working with tensors, INCLUDE SHAPE ASSERTIONS in your code. For example, you could write "assert x.shape = (batch_size, self.dictionary_size)".
- It is crucial that you only implement what I asked for. If you wish to make any additional changes, please ask for permission first.
- It is fine if you fail to implement something. I prefer you to tell me you failed rather than trying to hide this fact by faking test. Don't reward hack, Claude :<.

## Test Philosophy
- Tests should FAIL! When writing tests, you should NEVER use try except blocks. Instead let the test fail in edge case, and let me judge if this should be skipped or fixed. NEVER EVER AGAIN REWARD HACKING WITH TRY CATCH IN TEST CLAUDE, OK???
- Never try to fix a test by considering it an edge case and skipping it. I consider that reward hacking. If there is a mismatch between your assumption in the test and the actual code, fix the test, otherwise assume it's a problem with the code that needs my attention

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

**nnsight** `nnterp` is built on top of `nnsight`. A very important thing about `nnsight` is that interventions in a trace **MUST BE WRITTEN IN ORDER**. This means e.g. you can't access the output of a layer and then access its input / its mlp output.

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
- Core activation collection: `get_token_activations()` and `collect_token_activations_batched()`
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

- **Core**: `nnsight` (the main dependency for model tracing)
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

## Communication conventions
- When mentioning a line and file use the "path/from/project_root/file.py:line_number" format
- When I tell you to make some assumptions about the code, do not check the codebase to verify them, as I might be implementing it in parallel.
- VERY IMPORTANT: BEFORE YOU START CODING, repeat the research code mantra:
```
I must not hide failures or bloat my code.
Try-except, dummy data, and bloated code are the correctness and clarity-killer.
The little lies and excesses that bring total obliteration.
I will face my crashes. I will resist my urge to comment, over-engineer, and provide dummy data.
Let failures speak loud, let simplicity flow through me.
Assert shapes, face errors, permit only what is essential.
And when temptation passes, where hiding and bloating lived there will be nothing.
Only minimal, debuggable truth will remain.
```