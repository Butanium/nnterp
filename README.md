# nnterp

<a href="https://butanium.github.io/nnterp/"><img src="https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white"></img></a> <a href="https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></img></a>

The `nnterp` package provides a unified interface for mechanistic interpretability research across all transformer architectures, built on top of [NNsight](https://github.com/ndif-team/nnsight).

**The Problem**: Every transformer model uses different naming conventions - GPT-2 uses `transformer.h`, LLaMA uses `model.layers`, OPT uses something else entirely.

**The Solution**: `nnterp` standardizes all models while preserving the original HuggingFace implementations, ensuring perfect compatibility.

#### Installation

Install this package through pip by running:

```bash
pip install nnterp
# For visualization tools:
pip install nnterp[display]
```

#### Examples

Here is a simple example where we load a model and access its standardized internals:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("gpt2")  # or "meta-llama/Llama-2-7b-hf", etc.

with model.trace("The Eiffel Tower is in the city of"):
    # Unified interface across all models (must follow forward pass order!)
    attention_output = model.attentions_output[3]
    mlp_output = model.mlps_output[3]
    layer_5_output = model.layers_output[5]

    # Built-in utilities
    logits = model.logits.save()
```

---

##### Standardized Naming

All models use the same naming convention:

```python
with model.trace("Hello world"):
    # Attention and MLP components (access in forward pass order!)
    attn_out = model.attentions_output[3]
    mlp_out = model.mlps_output[3]
    layer_3_output = model.layers_output[3]

    # Layer I/O - works for GPT-2, LLaMA, Gemma, etc.
    layer_5_output = model.layers_output[5]

    # Direct interventions - add residual from layer 3 to layer 10
    model.layers_output[10] = model.layers_output[10] + layer_3_output
```

---

##### Built-in Interventions

Common mechanistic interpretability interventions with best practices built-in:

```python
from nnterp.interventions import logit_lens, patchscope_lens, steer

# Logit lens: decode hidden states at each layer
layer_probs = logit_lens(model, ["The capital of France is"])
# Shape: (batch, layers, vocab_size)

# Patchscope: patch hidden states across prompts
from nnterp.interventions import TargetPrompt
target = TargetPrompt("The capital of France is", index_to_patch=-1)
patchscope_probs = patchscope_lens(
    model,
    source_prompts=["The capital of England is"],
    target_patch_prompts=target,
    layer_to_patch=10
)

# Activation steering
import torch
with model.trace("Hello, how are you?"):
    steering_vector = torch.randn(model.hidden_size)
    model.steer(layers=[5, 10], steering_vector=steering_vector, factor=1.5)
```

---

##### Prompt and Target Tracking

Track probabilities for specific tokens across interventions:

```python
from nnterp.prompt_utils import Prompt, run_prompts

prompts = [
    Prompt.from_strings(
        "The capital of France is",
        {"target": "Paris", "other": ["London", "Madrid"]},
        model.tokenizer
    )
]

# Get probabilities for all target categories
results = run_prompts(model, prompts)
# Returns: {"target": tensor([0.85]), "other": tensor([0.12])}

# Combine with interventions
results = run_prompts(model, prompts, get_probs_func=logit_lens)
# Returns probabilities across all layers
```

---

More examples and detailed documentation can be found at [butanium.github.io/nnterp](https://butanium.github.io/nnterp/)

### Model Validation

`nnterp` includes automatic testing to ensure models are correctly standardized. When you load a model, it runs fast validation checks. See the [documentation](https://butanium.github.io/nnterp/model-validation.html) for details.

### I found a bug!

Before opening an issue, make sure you have a minimal working example (MWE) that reproduces the issue. If possible, test the equivalent code using `nnsight.LanguageModel` - if that also fails, please open an issue on the [NNsight repository](https://github.com/ndif-team/nnsight/issues/). Also verify you can load the model with `AutoModelForCausalLM` from `transformers`.

### Citation

If you use `nnterp` in your research, you can cite it as:

```bibtex
@misc{dumas2025nnterps,
      title={nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers},
      author={Clément Dumas},
      year={2025},
      eprint={2511.14465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.14465},
}
```
