# nnterp

## Installation
- `pip install nnterp`
- `pip install nnterp[display]` if you want to use the `display` module for visualizations

## Usage
### 1. Loading a Model

First, let's load a model in `nnsight` using `nnterp`'s `load_model` function.

```python
from nnterp import load_model

model_name = "meta-llama/Llama-2-7b-hf"
# Load the model (float16 and gpu by default)
nn_model = load_model(model_name)
tokenizer = nn_model.tokenizer
```

### 2. Creating and Running Prompts

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

### 3. Using Interventions

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

### 4. Using the Display Module
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