# nnterp

A Python package for neural network mechanistic interpretability using [nnsight](https://github.com/ndif-team/nnsight). This library provides tools for activation collection, interventions, and analysis of transformer models.

## Installation
- `pip install nnterp` - Basic installation
- `pip install nnterp[display]` - Includes visualization dependencies
- `pip install nnterp[vllm]` - Includes vLLM support for efficient inference (NOT supported yet)

## Key Features

### StandardizedTransformer
The `StandardizedTransformer` class provides a unified interface for working with different transformer architectures by standardizing module names and providing convenient accessors for intermediate activations.

```python
from nnterp import StandardizedTransformer

# Load a model with standardized naming
model = StandardizedTransformer("gpt2")

# Access layers and components using standardized names
with model.trace("Hello world"):
    # Direct layer access
    layer_0_input = model.layers_input[0].save()
    # Attention and MLP outputs
    attn_output = model.attentions_output[0].save()
    mlps_output = model.mlps_output[0].save()
    layer_0_output = model.layers_output[0].save()
    
    
    
    # Final logits and probabilities
    logits = model.logits.save()
    probs = model.next_token_probs.save()
```


### Collecting Activations

#### Single Batch Activation Collection
```python
from nnterp import get_token_activations

# Collect activations for all layers
prompt = "The quick brown fox jumps over the lazy dog"
activations = get_token_activations(nn_model, [prompt])

# Print activation shapes
for layer, activation in enumerate(activations):
    print(f"Layer {layer}: {activation.shape}")
```

#### Batched Activation Collection
For large datasets, use batched collection to optimize memory usage:

```python
from nnterp import collect_token_activations_batched

prompts = ["The quick brown fox", "jumps over the lazy dog"]
batch_size = 2
activations = collect_token_activations_batched(nn_model, prompts, batch_size)

for layer, activation in enumerate(activations):
    print(f"Layer {layer}: {activation.shape}")
```

### Prompt Utilities and Evaluation

```python
from nnterp import Prompt, run_prompts

# Create prompts with target tokens to track
prompts = [
    Prompt.from_strings("The quick brown fox", {"target": "jumps"}, tokenizer),
    Prompt.from_strings("Hello, how are you", {"target": "doing"}, tokenizer)
]

# Get target token probabilities
target_probs = run_prompts(nn_model, prompts, batch_size=2)

for prompt, probs in zip(prompts, target_probs["target"]):
    print(f"Prompt: {prompt.prompt}")
    print(f"Target Probabilities: {probs}")
```

### Interventions

#### Logit Lens
Analyze what the model "thinks" at each layer by projecting hidden states to vocabulary:

```python
from nnterp import logit_lens

prompt = "The quick brown fox jumps over the lazy dog"
logit_probs = logit_lens(nn_model, prompt)
print(f"Logit Lens Probabilities: {logit_probs.shape}")  # (num_prompts, num_layers, vocab_size)
```

#### Patchscope Lens
Compare activations between different contexts:

```python
from nnterp import patchscope_lens, TargetPrompt, repeat_prompt

source_prompt = "The quick brown fox"
target_prompt = TargetPrompt(prompt="jumps over the lazy dog", index_to_patch=-1)

# Or use the repeat prompt from the patchscopes paper
target_prompt = repeat_prompt(words=["king", "hello", "world"])

patchscope_probs = patchscope_lens(
    nn_model, 
    source_prompts=[source_prompt], 
    target_patch_prompts=[target_prompt]
)
```

#### Patchscope Generation
Generate text using patchscope interventions:

```python
from nnterp import patchscope_generate

generated_text = patchscope_generate(
    nn_model,
    prompts=["The capital of France is"],
    target_patch_prompt=TargetPrompt("Paris is the capital of", -1),
    max_length=20
)
```

#### Activation Steering
Steer model behavior using activation additions:

```python
from nnterp import steer

# Create a steering vector (e.g., from activation differences)
steering_vector = torch.randn(model.config.hidden_size)

with model.trace("Hello, how are you?"):
    # Apply steering at layer 10
    steer(model, layers=10, steering_vector=steering_vector, factor=2.0)
    output = model.lm_head.output.save()
```


### Visualization

```python
from nnterp.display import plot_topk_tokens

# Visualize top-k token probabilities across layers
fig = plot_topk_tokens(
    patchscope_probs,
    tokenizer,
    k=5,
    title="Patchscope Lens Probabilities",
    file="results.png",
    save_html=True
)
fig.show()
```

### Instruction-Tuned Models
For chat/instruction-tuned models, use specialized prompt functions:

```python
from nnterp.interventions import it_repeat_prompt

# Generate prompts adapted for instruction-tuned models
target_prompt = it_repeat_prompt(
    tokenizer,
    words=["Paris", "London", "Tokyo"],
    complete_prompt=True,
    add_user_instr=True
)
```

## Codebase Overview

- **`standardized_transformer.py`** - `StandardizedTransformer` class for unified model interface with standardized naming and convenient accessors
- **`nnsight_utils.py`** - Core utilities for model loading, activation collection, and model manipulation compatible with both TransformerLens and HuggingFace models
- **`interventions.py`** - Comprehensive intervention toolkit including:
  - Logit lens and patchscope lens
  - Activation steering and patching
  - Latent prompt manipulation
  - Advanced attention interventions
- **`prompt_utils.py`** - Utilities for creating and managing prompts with target token tracking
- **`display.py`** - Visualization tools for analyzing model outputs and probabilities

## Advanced Usage Example

```python
from nnterp import StandardizedTransformer, logit_lens, patchscope_lens, TargetPrompt
from nnterp.display import plot_topk_tokens
import torch as th

# Load model with standardized interface
model = StandardizedTransformer("gpt2")

# Analyze layer-wise predictions
prompt = "The quick brown fox jumps over the lazy dog"
logit_probs = logit_lens(model, prompt)

# Compare with patchscope
source_prompt = "The quick brown fox"
target_prompt = TargetPrompt(prompt="jumps over the lazy dog", index_to_patch=-1)
patchscope_probs = patchscope_lens(model, [source_prompt], [target_prompt])

# Visualize results
fig = plot_topk_tokens(
    logit_probs[0],  # First prompt
    model.tokenizer,
    k=5,
    title="Logit Lens Analysis",
    file="logit_lens.png"
)

# Layer-specific steering
steering_vector = th.randn(model.config.hidden_size) * 0.1  # Will be automatically moved to the different layers devices.
with model.trace("Hello, how are you?"):
    model.steer(layers=[10, 15], steering_vector=steering_vector, factor=1.5)
    steered_output = model.next_token_probs.save()
```

## Dependencies

- **Core**: `nnsight>=0.5.0`
- **Visualization**: `plotly`, `pandas` (install with `[display]` extra)
- **High-performance inference**: `vllm` (install with `[vllm]` extra) (NOT supported yet)

# Contributing
- Install the development environment with `make dev` or `uv sync --all-extras && uv pip install flash-attn --no-build-isolation`
- Create a git tag with the version number `git tag vx.y.z; git push origin vx.y.z`
- Build with `python -m build`
- Publish with e.g. `twine upload dist/*x.y.z*`
