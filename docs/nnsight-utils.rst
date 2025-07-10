NNsight Utils
=============

You can use nnterp utility functions with regular nnsight models, as long as they use llama or gemma naming conventions.

Setup
-----

.. code-block:: python

   from nnsight import LanguageModel
   from nnterp.nnsight_utils import (
       get_layer_output, get_attention_output, get_mlp_output,
       get_token_activations, collect_token_activations_batched,
       project_on_vocab, skip_layers
   )
   
   # Load model with proper renaming
   model = LanguageModel(
       "gpt2", 
       rename=dict(transformer="model", h="layers", ln_f="norm", attn="self_attn")
   )

Layer Access
------------

.. code-block:: python

   with model.trace("hello"):
       # Get layer outputs
       layer_5_out = get_layer_output(model, 5)
       
       # Get attention and MLP outputs
       attn_out = get_attention_output(model, 3)
       mlp_out = get_mlp_output(model, 3)

Projection and Utilities
-----------------------

.. code-block:: python

   with model.trace("The capital of France is"):
       hidden = get_layer_output(model, 5)
       logits = project_on_vocab(model, hidden)

Skip Layers
-----------

.. code-block:: python

   with model.trace("Hello world"):
       # Skip layers 2-5
       skip_layers(model, 2, 5)

Activation Collection
--------------------

.. code-block:: python

   # Single batch
   prompts = ["The capital of France is", "The weather today is"]
   with model.trace(prompts) as tracer:
       activations = get_token_activations(model, prompts, idx=-1, tracer=tracer)
       # Shape: (num_layers, batch_size, hidden_size)

   # Large datasets
   large_prompts = ["Sample text " + str(i) for i in range(100)]
   batch_activations = collect_token_activations_batched(
       model,
       large_prompts,
       batch_size=16,
       layers=[3, 9, 11],  # Only specific layers
       idx=-1  # Last token
   )

Requirements
-----------

These functions work with NNsight models that have been renamed to use llama/gemma conventions:

- ``model.layers`` (not ``transformer.h``)
- ``layers[i].self_attn`` (not ``layers[i].attn``)
- ``model.norm`` (not ``ln_f``)

For other architectures, use ``StandardizedTransformer`` which handles renaming automatically.