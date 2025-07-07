# %% [markdown]
# # Demo: nnterp Features Showcase
#
# This notebook demonstrates the key features of `nnterp`, which aims to offer a unified interface for all transformer models and give best `NNsight` practices for LLMs in everyone's hands.

# %% [markdown]
# ## 1. Standardized Interface
#
# Similar to [`transformer_lens`](https://github.com/TransformerLensOrg/TransformerLens), `nnterp` provides a standardized interface for all transformer models.
# The main difference is that `nnterp` still uses the huggingface implementation under the hood through `NNsight`, while transformer_lens uses its own implementation of the transformer architecture. However, each transformer implementation has its own quirks, such that `transformer_lens` is not able to support all models, and can sometimes have significant difference with the huggingface implementation.
#
# The way it's implemented is based on the `NNsight` built-in renaming feature, to make all models look like the llama naming convention, without having to write `model.model`, namely:
# ```ocaml
# StandardizedTransformer
# ├── layers
# │   ├── self_attn
# │   └── mlp
# ├── norm
# └── lm_head
# ```

# %%
from transformers import AutoModelForCausalLM

print(AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0"))
print(AutoModelForCausalLM.from_pretrained("gpt2"))

# %% [markdown]
# As you can see, the naming scheme of gpt2 is different from the llama naming convention.
# A simple way to fix this is to use the `rename` feature of `NNsight` to rename the gpt2 modules to the llama naming convention.

# %%
from nnsight import LanguageModel

model = LanguageModel(
    "gpt2", rename=dict(transformer="model", h="layers", ln_f="norm", attn="self_attn")
)
print(model)
# Access the attn module as if it was a llama model
print(model.model.layers[0].self_attn)

# %% [markdown]
# You can see the that renamed modules are displayed like `(new_name)/old_name`. However, many models family have their own naming convention, `nnterp` has a global renaming scheme that should transform any model to the llama naming convention. The easiest way to use it is to load your model using the `StandardizedTransformer` class that inherits from `nnsight.LanguageModel`.

# %%
from nnterp import StandardizedTransformer

# You will see the `layers` module printed two times, it'll be explained later.
nnterp_gpt2 = StandardizedTransformer("gpt2")
print(nnterp_gpt2)
# StandardizedTransformer also use `device_map="auto"` by default:
nnterp_gpt2.dispatch()
print(nnterp_gpt2.model.device)

# %% [markdown]
# Great! But I can see you at the back of the classroom, asking yourself:
# > "Why would you create a package that just pass the right dict to the `NNsight` `rename` feature?"
#
# And actually, I'm glad you asked! `StandardizedTransformer` and `nnterp` have a lot of other features, so bear with me!

# %% [markdown]
# ## 2. Accessing Modules I/O
# With `NNsight`, the most robust way to set the residual stream after layer 1 to be the residual stream after layer 0 for a LLama-like model would be:

# %%
llama = LanguageModel("Maykeye/TinyLLama-v0")
with llama.trace("hello"):
    llama.model.layers[1].output = (
        llama.model.layers[0].output[0],
        *llama.model.layers[1].output[1:],
    )

# %% [markdown]
# Note that the following can cause issues:

# %%
with llama.trace("hello"):
    # can't do this because .output is a tuple
    # llama.model.layers[1].output[0] = llama.model.layers[0].output[0]

    # Can cause errors with gradient computation
    llama.model.layers[1].output[0][:] = llama.model.layers[0].output[0]

with llama.trace("hello"):
    # Can cause errors with opt if you do this at its last layer (thanks pytest)
    llama.model.layers[1].output = (llama.model.layers[0].output[0],)

# %% [markdown]
# `nnterp` makes this much cleaner:

# %%
# First, you can access layer inputs and outputs directly:
with nnterp_gpt2.trace("hello"):
    # Access layer 5's output
    layer_5_output = nnterp_gpt2.layers_output[5]
    # Set layer 10's output to be layer 5's output
    nnterp_gpt2.layers_output[10] = layer_5_output

# You can also access attention and MLP outputs:
with nnterp_gpt2.trace("hello"):
    attn_output = nnterp_gpt2.attentions_output[3]
    mlp_output = nnterp_gpt2.mlps_output[3]

# %% [markdown]
# ## 3. Builtin interventions
#
# `StandardizedTransformer` also provides convenient methods for common operations:

# %%
import torch as th

# Project hidden states to vocabulary using the unembed norm and lm_head
with nnterp_gpt2.trace("The capital of France is"):
    hidden = nnterp_gpt2.layers_output[5]
    logits = nnterp_gpt2.project_on_vocab(hidden)

# Skip layers entirely
with nnterp_gpt2.trace("Hello world"):
    # Skip layer 1
    nnterp_gpt2.skip_layer(1)
    # Skip layers 2 through 3 (inclusive)
    nnterp_gpt2.skip_layers(2, 3)

# This is useful if you want to start at a later layer than the first one
with nnterp_gpt2.trace("Hello world") as tracer:
    layer_6_out = nnterp_gpt2.layers_output[6].save()
    tracer.stop()  # avoid computations after layer 6

with nnterp_gpt2.trace("Hello world"):
    nnterp_gpt2.skip_layers(0, 6, skip_with=layer_6_out)
    half_half_logits = nnterp_gpt2.logits.save()

with nnterp_gpt2.trace("Hello world"):
    vanilla_logits = nnterp_gpt2.logits.save()

assert th.allclose(vanilla_logits, half_half_logits)  # they should be the same

# Direct steering
steering_vector = th.randn(768)  # gpt2 hidden size
with nnterp_gpt2.trace("The weather today is"):
    nnterp_gpt2.steer(layers=[1, 3], steering_vector=steering_vector, factor=0.5)

# %% [markdown]
# ## 4. Attention Probabilities
#
# For models that support it, you can access attention probabilities directly:

# %%
nnterp_gpt2.tokenizer.padding_side = (
    "left"  # ensure left padding for easy access to the first token
)

with th.no_grad():
    with nnterp_gpt2.trace("The cat sat on the mat"):
        # Access attention probabilities for layer 5
        attn_probs_l2 = nnterp_gpt2.attention_probabilities[2].save()
        attn_probs = nnterp_gpt2.attention_probabilities[5].save()
        print(
            f"Attention probs shape will be: (batch, heads, seq_len, seq_len): {attn_probs.shape}"
        )
        # knock out the attention to the first token
        attn_probs[:, :, :, 0] = 0
        attn_probs /= attn_probs.sum(dim=-1, keepdim=True)
        corr_logits = nnterp_gpt2.logits.save()
    with nnterp_gpt2.trace("The cat sat on the mat"):
        baseline_logits = nnterp_gpt2.logits.save()

assert not th.allclose(corr_logits, baseline_logits)

sums = attn_probs_l2.sum(dim=-1)
assert th.allclose(sums, th.ones_like(sums))

# %% [markdown]
# Under the hood this uses the new tracing system implemented in `NNsight v0.5` which allow to access most model intermediate variables during the forward pass. This means that if the `transformers` implementation were to change, this could break or give unexpected results, so it is recommended to use one of the tested versions of `transformers` and to check that the attention probabilities hook makes sense by calling `model.attention_probabilities.print_source()` if you want to use a different version of `transformers` / a architecture that has not been tested.

# %%
nnterp_gpt2.attention_probabilities.print_source()  # pretty markdown display in a notebook

# %% [markdown]
# ## 5. Activation Collection
#
# `nnterp` provides utilities for collecting activations efficiently:

# %%
from nnterp.nnsight_utils import (
    get_token_activations,
    collect_token_activations_batched,
)

# Collect activations for specific tokens
prompts = ["The capital of France is", "The weather today is"]
with nnterp_gpt2.trace(prompts) as tracer:
    # Get last token activations for all layers
    activations = get_token_activations(nnterp_gpt2, prompts, idx=-1, tracer=tracer)
    # activations shape: (num_layers, batch_size, hidden_size)

# For large datasets, use batched collection
large_prompts = ["Sample text " + str(i) for i in range(100)]
batch_activations = collect_token_activations_batched(
    nnterp_gpt2,
    large_prompts,
    batch_size=16,
    layers=[3, 9, 11],  # Only collect specific layers, default is all layers
    idx=-1,  # Last token (default)
)
print(f"Batched activations shape: {batch_activations.shape}")

# %% [markdown]
# ## 6. Prompt Utilities
#
# `nnterp` provides utilities for working with prompts and tracking probabilities of first tokens of certain strings. It tracks both the first token of "string" and " string".
#
# You can provide multiple string per category, the probabilities returned will be the sum of the probabilities of all the first tokens of the strings.

# %%
from nnterp.prompt_utils import Prompt, run_prompts

# Create prompts with target tokens to track
prompt1 = Prompt.from_strings(
    "The capital of France (not England or Spain) is",
    {
        "target": "Paris",
        "traps": ["London", "Madrid"],
        "longstring": "the country of France",
    },
    nnterp_gpt2.tokenizer,
)
for name, tokens in prompt1.target_tokens.items():
    print(f"{name}: {nnterp_gpt2.tokenizer.convert_ids_to_tokens(tokens)}")

prompt2 = Prompt.from_strings(
    "The largest planet (not Earth or Neptune) is",
    {"target": "Jupiter", "traps": ["Earth", "Neptune"], "longstring": "Palace planet"},
    nnterp_gpt2.tokenizer,
)
for name, tokens in prompt2.target_tokens.items():
    print(f"{name}: {nnterp_gpt2.tokenizer.convert_ids_to_tokens(tokens)}")

# Run prompts and get probabilities for target tokens
results = run_prompts(nnterp_gpt2, [prompt1, prompt2], batch_size=2)
print("Target token probabilities:")
for target, probs in results.items():
    print(f"  {target}: shape {probs.shape}")

# %% [markdown]
# ## 7. Interventions
#
# `nnterp` provides several intervention methods inspired by mechanistic interpretability research:

# %%
from nnterp.interventions import (
    logit_lens,
    patchscope_lens,
    TargetPrompt,
    repeat_prompt,
    steer,
)

# Logit Lens: See predictions at each layer
prompts = ["The capital of France is", "The sun rises in the"]
probs = logit_lens(nnterp_gpt2, prompts)
print(f"Logit lens output shape: {probs.shape}")  # (batch, layers, vocab)

# Patchscope: Replace activations from one context into another
source_prompts = ["Paris is beautiful", "London is foggy"]
custom_target_prompt = TargetPrompt("city: Paris\nfood: croissant\n?", -1)
target_prompt = repeat_prompt()  # Creates a repetition task
custom_repeat_prompt = repeat_prompt(
    words=["car", "cross", "azdrfa"],
    rel=":",
    sep="\n\n",
    placeholder="*",
)
print(f"repeat_prompt: {custom_repeat_prompt}")
print(f"custom_repeat_prompt: {custom_repeat_prompt}")
patchscope_probs = patchscope_lens(
    nnterp_gpt2, source_prompts=source_prompts, target_patch_prompts=target_prompt
)
print(f"patchscope_probs: {patchscope_probs.shape}")

# Steering with intervention function
with nnterp_gpt2.trace("The weather is"):
    steer(nnterp_gpt2, layers=[5, 10], steering_vector=steering_vector)

# %% [markdown]
# You can use a combination of run_prompts and interventions to get the probabilities of certain tokens according to your custom intervention.
# %%
demo_model = StandardizedTransformer("google/gemma-2-2b")
# uncomment if you don't have a GPU
# demo_model = nnterp_gpt2

prompts_str = [
    "The translation of 'car' in French is",
    "The translation of 'cat' in Spanish is",
]
tokens = [
    {"target": ["voiture", "bagnole"], "english": "car", "format": "'"},
    {"target": ["gato", "minino"], "english": "cat", "format": "'"},
]
prompts = [
    Prompt.from_strings(prompt, tokens, demo_model.tokenizer)
    for prompt, tokens in zip(prompts_str, tokens)
]
results = run_prompts(demo_model, prompts, batch_size=2, get_probs_func=logit_lens)
for category, probs in results.items():
    print(f"{category}: {probs.shape}")  # (batch, layers)

# Create a plotly plot showing mean probabilities for each category across layers
import plotly.graph_objects as go

# Calculate mean probabilities across batches for each category and layer
mean_probs = {category: probs.mean(dim=0) for category, probs in results.items()}

fig = go.Figure()

# Add a line for each category
for category, probs in mean_probs.items():
    fig.add_trace(
        go.Scatter(
            x=list(range(len(probs))),
            y=probs.tolist(),
            mode="lines+markers",
            name=category,
            line=dict(width=2),
            marker=dict(size=6),
        )
    )

fig.update_layout(
    title="Mean Token Probabilities Across Layers",
    xaxis_title="Layer",
    yaxis_title="Mean Probability",
    hovermode="x unified",
    template="plotly_white",
)

fig.show()


# %% [markdown]
# ## 8. Visualization
#
# Finally, `nnterp` provides visualization utilities for analyzing model probabilities and prompts:

# %%
from nnterp.display import plot_topk_tokens, prompts_to_df

probs = logit_lens(demo_model, prompts_str[0])
# Visualize top tokens from logit lens
plot_topk_tokens(
    probs[0],
    demo_model.tokenizer,
    k=5,
    width=1000,
    height=1000,
    title="Top 5 tokens at each layer for 'The translation of 'car' in French is",
)

# Convert prompts to DataFrame for analysis
df = prompts_to_df(prompts, demo_model.tokenizer)
print("\nPrompts DataFrame:")
display(df)

# %% [markdown]
# ## Summary
#
# `nnterp` provides a unified, standardized interface for working with transformer models, built on top of `nnsight`. Key features include:
#
# 1. **Standardized naming** across all transformer architectures
# 2. **Easy access** to layer/attention/MLP inputs and outputs
# 3. **Built-in methods** for common operations (steering, skipping layers, projecting to vocab)
# 4. **Efficient activation collection** with batching support
# 5. **Prompt utilities** for tracking target tokens
# 6. **Intervention methods** from mechanistic interpretability research
# 7. **Visualization tools** for analyzing model behavior
#
# All of this while maintaining the full power and flexibility of `nnsight` under the hood!

# %% [markdown]
# # Appendix: `NNsight` cheatsheet

# %% [markdown]
# ## 1) You must execute your interventions in order
# In the new `NNsight` versions, it is enforced that you must access to model internals *in the same order* as the model execute them.

# %%
from nnterp import StandardizedTransformer

nnterp_gpt2 = StandardizedTransformer("gpt2")

with nnterp_gpt2.trace("My tailor is rich"):
    l2 = nnterp_gpt2.layers_output[2]
    l1 = nnterp_gpt2.layers_output[1]  # will fail! You need to collect l1 before l2

# %% [markdown]
# ## 2) Gradient computation
# To compute gradients, you need to open a `.backward()` context, and save the gradients *inside it*.

# %%
with nnterp_gpt2.trace("My tailor is rich"):
    l1_out = nnterp_gpt2.layers_output[1]  # get l1 before accessing logits
    logits = nnterp_gpt2.output.logits
    with logits.sum().backward(
        retain_graph=True
    ):  # use retain_graph if you want to do multiple backprops
        if False:
            l1_grad = nnterp_gpt2.layers_output[1].grad.save()
            # this would fail as we'd access nnterp_gpt2.layers_output[1] after nnterp_gpt2.output
        l1_grad = l1_out.grad.save()
    with (logits.sum() ** 2).backward():
        l1_grad_2 = l1_out.grad.save()

assert not th.allclose(l1_grad, l1_grad_2)

# %% [markdown]
# ## 3) Use tracer.stop() to save useless computations
# If you're just computing activations, don't forget to call `tracer.stop()` at the end of your trace. This will stop the model from executing the rest of its computations, and save you some time, as demonstrated below (with the contribution of Claude 4 Sonnet).

# %%
import time
import pandas as pd

print(
    "🎭 Welcome to the Theatrical Performance Comparison! 🎭\n"
    + "=" * 60
    + "\n\n🐌 ACT I: 'The Tragedy of the Unstoppable Tracer' 🐌\nIn which our hero forgets to call tracer.stop()..."
)

start_time = time.time()
for _ in range(30):
    with nnterp_gpt2.trace(["Neel Samba", "Chris Aloha"]):
        out5 = nnterp_gpt2.layers_output[5].save()
end_time = time.time()
nostop_time = end_time - start_time

print(
    f"⏰ Duration of suffering: {nostop_time:.4f} seconds\n\n⚡ ACT II: 'The Redemption of the Stopped Tracer' ⚡\nOur hero learns the ancient art of tracer.stop()..."
)


start_time = time.time()
for _ in range(30):
    with nnterp_gpt2.trace(["Neel Samba", "Chris Aloha"]) as tracer:
        out5 = nnterp_gpt2.layers_output[5].save()
        tracer.stop()
end_time = time.time()
stop_time = end_time - start_time

print(f"⏰ Duration of enlightenment: {stop_time:.4f} seconds")

speedup = nostop_time / stop_time
time_saved = nostop_time - stop_time

# fun display
print("\n" + "=" * 60 + "\n🎉 THE GRAND RESULTS SPECTACULAR! 🎉\n" + "=" * 60)
results_df = pd.DataFrame(
    {
        "🎭 Performance Type": [
            "Without tracer.stop() 🐌",
            "With tracer.stop() ⚡",
            "Time Saved 💰",
        ],
        "⏱️ Time (seconds)": [
            f"{nostop_time:.4f}",
            f"{stop_time:.4f}",
            f"{time_saved:.4f}",
        ],
        "🎯 Rating": ["Tragic 😭", "Magnificent! 🌟", "PROFIT! 📈"],
    }
)
display(results_df)
speedup_bars = int(speedup * 10)
meter = "█" * min(speedup_bars, 48) + "░" * (50 - min(speedup_bars, 48))
print(
    f"\n🏎️ SPEEDUP METER 🏎️\n┌{'─' * 50}┐\n│{meter}│\n└{'─' * 50}┘\n   💫 COSMIC SPEEDUP: {speedup:.2f}x FASTER! 💫"
)


# %% [markdown]
# ## 4) Using NNsight builtin cache to collect activations
#
# `NNsight 0.5` introduces a builtin way to cache activations during the forward pass. Be careful not to call `tracer.stop()` before all the module of the cache have been accessed.
#
# NOTE: Currently doesn't work with renamed names.

# %%
with nnterp_gpt2.trace("Hello") as tracer:
    cache = tracer.cache(modules=[layer for layer in nnterp_gpt2.layers[::2]]).save()

print(cache.keys())
print(cache["model.transformer.h.10"].output)
