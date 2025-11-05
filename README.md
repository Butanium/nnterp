Try the demo: [![Try Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb) / read [the doc](https://butanium.github.io/nnterp/)

# nnterp

**A unified interface for all transformer models that puts best `NNsight` practices for LLMs in everyone's hands.**

Built on top of [NNsight](https://github.com/ndif-team/nnsight), `nnterp` provides a standardized interface for mechanistic interpretability research across all transformer architectures. Unlike `transformer_lens` which reimplements transformers, `nnterp` preserves the original HuggingFace implementations while solving the naming convention chaos through intelligent renaming.

## Why nnterp?

**The Problem**: Every transformer model uses different naming conventions - GPT-2 uses `transformer.h`, LLaMA uses `model.layers`, OPT uses something else entirely. This makes mechanistic interpretability research painful as you can't just change the name of the model and expect the rest of your code to work.

**The Solution**: `nnterp` standardizes all models to use something close to the LLaMA naming convention:
```
StandardizedTransformer
├── layers
│   ├── self_attn
│   └── mlp
├── ln_final
└── lm_head
```
and include built-in properties like `model.logits` and `model.next_token_probs`.

Unlike other libraries that reimplement transformers, `nnterp` uses `NNsight`'s renaming feature to work with the original HuggingFace implementations, ensuring perfect compatibility and preventing subtle bugs.

## Installation
- `pip install nnterp` - Basic installation
- `pip install nnterp[display]` - Includes visualization dependencies
<!-- - `pip install nnterp[vllm]` - Includes vLLM support for efficient inference (NOT supported yet) -->

## Key Features

### 1. StandardizedTransformer - One Interface, All Models
The `StandardizedTransformer` class provides a unified interface across all transformer architectures, with automatic device mapping and standardized naming.

```python
from nnterp import StandardizedTransformer

# Load any model with standardized naming (with device_map="auto" by default)
model = StandardizedTransformer("gpt2")  # or "meta-llama/Llama-2-7b-hf", "google/gemma-2b", etc.

# Clean, intuitive access to model internals
with model.trace("Hello world"):
    # Direct attention/MLP access
    attn_output = model.attentions_output[3]
    mlp_output = model.mlps_output[3]

    # Easy layer I/O access
    layer_5_output = model.layers_output[5]
    layer_10_output = model.layers_output[10] = layer_5_output  # Skip layers 6-10
    
    
    # Built-in utilities
    logits = model.project_on_vocab(layer_5_output)  # Project to vocabulary
    model.steer(layers=[1, 3], steering_vector=vector, factor=0.5)  # Activation steering
```


### 2. Advanced Attention Access
Access and modify attention probabilities directly (requires `enable_attention_probs=True`):

```python
# Load model with attention probabilities enabled
model = StandardizedTransformer("gpt2", enable_attention_probs=True)

# Access attention probabilities during forward pass
with model.trace("The cat sat on the mat"):
    attn_probs = model.attention_probabilities[5].save()  # Layer 5 attention

    # Modify attention patterns
    attn_probs[:, :, :, 0] = 0  # Zero out attention to first token
    attn_probs /= attn_probs.sum(dim=-1, keepdim=True)  # Renormalize

    modified_logits = model.logits.save()

# Check what transformations are applied
model.attention_probabilities.print_source()
```

### 3. Efficient Activation Collection
Collect activations for a certain token position. (For all token use `NNsight` cache functionality):

```python
from nnterp.nnsight_utils import get_token_activations, collect_token_activations_batched

# Single batch - automatically uses tracer.stop() for efficiency
prompts = ["The capital of France is", "The weather today is"]
activations = get_token_activations(model, prompts, idx=-1)  # Last token
# Shape: (num_layers, batch_size, hidden_size)

# Large-scale batched collection with memory optimization
large_prompts = ["Sample text " + str(i) for i in range(1000)]
batch_activations = collect_token_activations_batched(
    model, large_prompts, batch_size=16, layers=[3, 9, 11]  # Specific layers only
)
```

### 4. Prompt and completion tracking
Track probabilities for specific target tokens in the answer of a prompt. This will automatically track both the first token of "target string" and " target string" (with a space).

```python
from nnterp.prompt_utils import Prompt, run_prompts

# Create prompts with target tokens to track
# Automatically handles both "word" and " word" tokenization
prompts = [
    Prompt.from_strings(
        "The capital of France (not England or Spain) is",
        {
            "target": "Paris",
            "traps": ["London", "Madrid"],  # Track competing options
            "concept": ["France", "country"]  # Track related concepts
        },
        model.tokenizer
    )
]

# Get probabilities for all target categories
results = run_prompts(model, prompts, batch_size=2)
for category, probs in results.items():
    print(f"{category}: {probs.shape}")  # Shape: (batch_size,)
```

### 5. Mechanistic Interpretability Interventions
All the classic interventions with best practices built-in:

#### Logit Lens
```python
from nnterp.interventions import logit_lens

prompts = ["The capital of France is", "The sun rises in the"]
layer_probs = logit_lens(model, prompts)  # (batch, layers, vocab_size)

# Combine with target tracking for precise analysis
results = run_prompts(model, prompts, get_probs_func=logit_lens)
# Returns probabilities for each target category across all layers
```

#### Patchscope
```python
from nnterp.interventions import patchscope_lens, TargetPrompt, repeat_prompt

# Custom target prompt
target_prompt = TargetPrompt("The capital of France is", index_to_patch=-1)

# Or use the repeat prompt from the patchscopes paper
target_prompt = repeat_prompt(words=["king", "hello", "world"])

source_prompts = ["The capital of France is", "The capital of England is"]
patchscope_probs = patchscope_lens(
    model, source_prompts=source_prompts, target_patch_prompts=target_prompt
)
```

#### Activation Steering
```python
from nnterp.interventions import steer
import torch

# Create steering vector (e.g., from activation differences between concepts)
steering_vector = torch.randn(model.config.hidden_size)

with model.trace("The weather today is"):
    # Apply steering at multiple layers
    steer(model, layers=[5, 10], steering_vector=steering_vector, factor=1.5)
    steered_output = model.logits.save()

# Or use the built-in method
with model.trace("Hello, how are you?"):
    model.steer(layers=[1, 3], steering_vector=steering_vector, factor=0.5)
```


### 6. A few visualization tools
Built-in plotting for immediate insights:

```python
from nnterp.display import plot_topk_tokens, prompts_to_df

# Visualize top-k token probabilities across layers
probs = logit_lens(model, "The capital of France is")
fig = plot_topk_tokens(
    probs[0],  # First prompt
    model.tokenizer,
    k=5,
    title="Layer-wise Token Predictions",
    width=1000,
    height=600
)
fig.show()

# Convert prompts to DataFrame for analysis
df = prompts_to_df(prompts, model.tokenizer)
print(df.head())
```

## Complete Research Example

```python
from nnterp import StandardizedTransformer
from nnterp.prompt_utils import Prompt, run_prompts
from nnterp.interventions import logit_lens
from nnterp.display import plot_topk_tokens
import torch
import plotly.graph_objects as go

# Load any model with unified interface
model = StandardizedTransformer("google/gemma-2-2b")

# Create research prompts with target tracking
prompts = [
    Prompt.from_strings(
        "The translation of 'car' in French is",
        {"target": ["voiture", "bagnole"], "english": "car"},
        model.tokenizer
    ),
    Prompt.from_strings(
        "The translation of 'cat' in Spanish is", 
        {"target": ["gato", "minino"], "english": "cat"},
        model.tokenizer
    )
]

# Run analysis with logit lens intervention
results = run_prompts(model, prompts, batch_size=2, get_probs_func=logit_lens)

# Create publication-ready visualization
mean_probs = {category: probs.mean(dim=0) for category, probs in results.items()}
fig = go.Figure()
for category, probs in mean_probs.items():
    fig.add_trace(go.Scatter(
        x=list(range(len(probs))), y=probs.tolist(),
        mode="lines+markers", name=category
    ))
fig.update_layout(
    title="Target Token Emergence Across Layers",
    xaxis_title="Layer", yaxis_title="Mean Probability"
)
fig.show()

# Advanced interventions
with model.trace("The weather today is"):
    # Skip early layers
    model.skip_layers(0, 5)
    
    # Apply steering
    steering_vector = torch.randn(model.config.hidden_size)
    model.steer(layers=[10, 15], steering_vector=steering_vector, factor=1.2)
    
    # Project intermediate states
    layer_20_logits = model.project_on_vocab(model.layers_output[20])
    final_logits = model.logits.save()
```


## Automatic Testing & Validation

nnterp includes comprehensive automatic testing to prevent silent failures. When you load a model, fast tests automatically run to ensure:

- **Model renaming correctness**: All modules are properly renamed to the standardized interface
- **Module output shapes**: Layer outputs and attention probabilities have the expected shapes
- **Attention probabilities**: Attention at a token position sums to 1

Also, when a new version of `nnterp` is released, tests are ran on most model architectures. Those tests are included in the package, and `nnterp` will warn you if the architecture you're using was tested, and if any of the tests failed. If you use a different version of `NNsight` or `transformers` or another model architecture, you can run the tests manually:

```bash
# Test specific models
python -m nnterp run_tests --model-names "gpt2" "meta-llama/Llama-2-7b-hf"

# Test using toy models (if available)
python -m nnterp run_tests --class-names "LlamaForCausalLM" "GPT2LMHeadModel"
```

## Dependencies
The `transformers` version is pinned to `4.53.x` as it's the only one that was tested for now. When a model is loaded in `StandardizedTransformer`, it will go through a series of checks to make sure that the model is still compatible.

- **Core**: `nnsight>=0.5.0`, `transformers==4.53.x`
- **Visualization**: `plotly`, `pandas` (install with `[display]` extra)
<!-- - **High-performance inference**: `vllm` (install with `[vllm]` extra) (NOT supported yet) -->

# I found a bug!
Before opening an issue, make sure that you have a MWE (minimal working example) that reproduces the issue, and if possible, the equivalent code using `NNsight.LanguageModel`. If the NNsight MWE also fails, please open an issue on the [NNsight repository](https://github.com/ndif-team/nnsight/issues/). Also make sure that you can load the model with `AutoModelForCausalLM` from `transformers`.

## Known issues:
- You might encounter cryptic errors of the form `With block not found at line xyz`. In this case, restart your Notebook, and if it doesn't fix it delete your python cache e.g.  using 
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + ; find . -name "*.pyc" -delete ; find . -name "*.pyo" -delete 
```

# Contributing
Contribution are welcome! If a functionality is missing, and you implemented it for your reasearch, please open a PR, so that people in the community can benefit from it. That include adding support for new models with custom renamings!

## Next steps
Here are some nice features that could be cool to have, and for which I'd be happy to accept PRs (ordered by most to least useful imo):
- [ ] Add helpers for getting gradients
- [ ] Add support for `vllm` when `NNsight` supports it
- [ ] Add helpers for `NNsight`'s cache as it returns raw tuple outputs instead of nice vectors.
- [ ] Add access to k/q/v
- [ ] Fix typing

## Development
- Install the development environment with `make dev` or `uv sync --all-extras`. Add `uv pip install flash-attn --no-build-isolation` to support models like `Phi` that require `flash-attn`.
- Install pre-commit hooks with `pre-commit install` to automatically update `docs/llms.txt` when modifying RST files and format the code with `black`.
You might encounter the error `with block not found at line xyz` when running the tests. In this case run `make clean` to remove the python cache and try again (NOTE: this should be fixed in latest `NNsight` versions).
- Create a git tag with the version number `git tag vx.y.z; git push origin vx.y.z`
- Build with `python -m build`
- Publish with e.g. `twine upload dist/*x.y.z*`
- test with `pytest --cache-clear`. **cache-clear is mandatory for now otherwise `NNsight`'s source can break.** It might not be sufficient, in which case you can do `make clean` to remove Python cache.


# Citation
If you use `nnterp` in your research, you can cite it as:

```bibtex
@inproceedings{
dumas2025nnterp,
title={nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers},
author={Cl{\'e}ment Dumas},
booktitle={Mechanistic Interpretability Workshop at NeurIPS 2025},
year={2025},
url={https://openreview.net/forum?id=ACic3VDIHp}
}
```