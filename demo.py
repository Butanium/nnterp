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
# The way it's implemented is based on the `NNsight` built-in renaming feature, to make all models look like the llama naming convention, namely:
# ```
# StandardizedTransformer
# ├── model
# │   ├── layers
# │   │   ├── self_attn
# │   │   └── mlp
# │   └── norm
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
model = LanguageModel("gpt2", rename=dict(transformer="model", h="layers", ln_f="norm"))
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

# %% [markdown]
# Great! But I can see you at the back of the classroom, asking yourself:
# > "Why would you create a package that just pass the right dict to the `NNsight` `rename` feature?"
#
# And actually, I'm glad you asked! `StandardizedTransformer` has a lot of other features, so bear with me!

# %% [markdown]
# ## 2. Accessing Modules I/O
# With `NNsight`, the most robust way to set the residual stream after layer 1 to be the residual stream after layer 0 would be:
# ```py
# model.model.layers[1].output = (model.model.layers[0].output[0], *model.model.layers[1].output[1:])
# ```
# Note that the following can cause issues:
# ```py
# # can't do this because output is a tuple
# model.model.layers[1].output[0] = model.model.layers[0].output[0]
# 
# # Can cause errors with gradient computation
# model.model.layers[1].output[0][:] = model.model.layers[0].output[0]
# 
# # Can cause errors with opt if you do this at its last layer (thanks pytest)
# model.model.layers[1].output = (model.model.layers[0].output, )
# ```
# 
# `nnterp` makes this much cleaner:

# %%
# First, you can 
with nnterp_gpt2.trace("hello"):
    nnterp_gpt2.layers[]

