# nnterp

## Installation
- `pip install nnterp`
- `pip install nnterp[display]` if you want to use the `display` module for visualizations

## Usage
### Loading a Model

First, let's load a model in `nnsight` using `nnterp`'s `load_model` function.

```python
from nnterp import load_model

model_name = "meta-llama/Llama-2-7b-hf"
# Load the model (float16 and gpu by default)
nn_model = load_model(model_name)
tokenizer = nn_model.tokenizer
```

### Collecting activations

To collect activations from a model using `nnterp`, you can use the `collect_activations` function. This function takes the following parameters:

- `nn_model`: The NNSight model.
- `prompts`: The prompts for which you want to collect activations.
- `layers`: The layers for which you want to collect activations. If not specified, activations will be collected for all layers.
- `get_activations`: A function to get the activations. By default, it will collect the layer output, but you can also collect other things like attention input/output
- `remote`: Whether to run the model on the remote device.
- `idx`: The index of the token to collect activations for.
- `open_context`: Whether to open a context for the model trace. You can set to false if you want to collect activations in an already opened nnsight tracing context.

Here's an example usage of `collect_activations`:

```python
# Load the model
nn_model = load_model(model_name)

# Create a prompt
prompt = "The quick brown fox jumps over the lazy dog"

# Collect activations for all layers
activations = collect_activations(nn_model, [prompt])

# Print the activations
for layer, activation in enumerate(activations):
    print(f"Layer {layer}: {activation.shape}")
```

### Collecting activations in batches

If you have a large number of prompts and want to collect activations in batches to optimize memory usage, you can use the `collect_activations_batched` function. This function has similar parameters to `collect_activations`, but also takes a `batch_size` parameter to specify the batch size for collecting activations.

Here's an example usage of `collect_activations_batched`:

```python
# Load the model
nn_model = load_model(model_name)

# Create a list of prompts
prompts = ["The quick brown fox", "jumps over the lazy dog"]

# Collect activations in batches
batch_size = 2
activations = collect_activations_batched(nn_model, prompts, batch_size)

# Print the activations
for layer, activation in enumerate(activations):
    print(f"Layer {layer}: {activation}")
```

### Creating and Running Prompts

Next, we create some toy prompts and run them through the model to get the next token probabilities.

```python
from nnterp.prompt_utils import Prompt, run_prompts

# Create toy prompts
prompts = [
    Prompt.from_strings("The quick brown fox", {"target": "jumps"}, tokenizer),
    Prompt.from_strings("Hello, how are you", {"target": "doing"}, tokenizer)
]

# Run prompts through the model
target_probs = run_prompts(nn_model, prompts, batch_size=2)

# Print the results
for prompt, probs in zip(prompts, target_probs["target"]):
    print(f"Prompt: {prompt.prompt}")
    print(f"Target Probabilities: {probs}")
```

### Using Interventions

Now, let's use some interventions like `logit_lens`

#### Logit Lens

```python
from nnterp.interventions import logit_lens

# Create a toy prompt
prompt = "The quick brown fox jumps over the lazy dog"

# Get the logit lens probabilities
logit_probs = logit_lens(nn_model, prompt)

# Print the results
print(f"Logit Lens Probabilities: {logit_probs}")
```

#### Patchscope Lens

```python
from nnterp.interventions import patchscope_lens, TargetPrompt

# Create source and target prompts
source_prompt = "The quick brown fox"
target_prompt = TargetPrompt(prompt="jumps over the lazy dog", index_to_patch=-1)

# Get the patchscope lens probabilities
patchscope_probs = patchscope_lens(nn_model, source_prompts=[source_prompt], target_patch_prompts=[target_prompt])

# Print the results
print(f"Patchscope Lens Probabilities: {patchscope_probs}")
```

### Using the Display Module
from nnterp.display import plot_topk_tokens

```python
# Plot Patchscope Lens Probabilities and save the figure to test.png and test.html
fig = plot_topk_tokens(
    patchscope_probs,
    tokenizer,
    k=5,
    title="Patchscope Lens Probabilities",
    file="test.png",
    save_html=True,  # Default is True
)
fig.show()

### Full Example

Here is a full example combining all the above functionalities:

```python
from nnterp import load_model
from nnterp.prompt_utils import Prompt, run_prompts
from nnterp.interventions import logit_lens, patchscope_lens, TargetPrompt

# Load the model
model_name = "meta-llama/Llama-2-7b-hf"
nn_model = load_model(model_name, trust_remote_code=False, device_map="auto")
tokenizer = nn_model.tokenizer

# Create toy prompts
prompts = [
    Prompt.from_strings("The quick brown fox", {"target": "jumps"}, tokenizer),
    Prompt.from_strings("Hello, how are you", {"target": "doing"}, tokenizer)
]

# Run prompts through the model
target_probs = run_prompts(nn_model, prompts, batch_size=2)

# Print the results
for prompt, probs in zip(prompts, target_probs["target"]):
    print(f"Prompt: {prompt.prompt}")
    print(f"Target Probabilities: {probs}")

# Logit Lens
prompt = "The quick brown fox jumps over the lazy dog"
logit_probs = logit_lens(nn_model, prompt)
print(f"Logit Lens Probabilities: {logit_probs}")

# Patchscope Lens
source_prompt = "The quick brown fox"
target_prompt = TargetPrompt(prompt="jumps over the lazy dog", index_to_patch=-1)
patchscope_probs = patchscope_lens(nn_model, source_prompts=[source_prompt], target_patch_prompts=[target_prompt])
print(f"Patchscope Lens Probabilities: {patchscope_probs}")
```

## Codebase Overview
- `nnsight_utils.py` basically allows you to deal with TL and HF models in a similar way.
- `interventions.py` is a module that contains tools like logit lens, patchscope lens and other interventions.
- `prompt_utils.py` contains utils to create prompts for which you want to track specific tokens in the next token distribution and run interventions on them and collect the probabilities of the tokens you're interested in.

# Contributing
- Create a git tag with the version number `git tag vx.y.z; git push origin vx.y.z`
- Build with `python -m build`
- Publish with e.g. `twine upload dist/*x.y.z*`