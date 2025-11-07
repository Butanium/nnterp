Interventions
=============

.. meta::
   :llm-description: Common intervention methods for transformer analysis. Includes logit lens for layer-wise predictions and patchscope for cross-context activation replacement.

Analysis Methods
----------------

Logit Lens
~~~~~~~~~~

See predictions at each layer:

.. code-block:: python

   from nnterp.interventions import logit_lens
   
   prompts = ["The capital of France is", "The sun rises in the"]
   probs = logit_lens(model, prompts)
   # Shape: (batch, layers, vocab)

Return inverse logits (projection on negated normalized hidden states), computed as softmax(lm_head(-ln_final(hidden_states))):

.. code-block:: python

   probs, inv_probs = logit_lens(model, prompts, return_inv_logits=True)
   # Both tensors shape: (batch, layers, vocab)
   # inv_probs computed as softmax(lm_head(-ln_final(hidden_states)))

Patchscope
~~~~~~~~~~

Replace activations from one context into another:

.. code-block:: python

   from nnterp.interventions import patchscope_lens, TargetPrompt, repeat_prompt
   
   source_prompts = ["Paris is beautiful", "London is foggy"]
   target_prompt = TargetPrompt("city: Paris\\nfood: croissant\\n?", -1)
   
   # Or use repeat_prompt with custom index_to_patch
   target_prompt = repeat_prompt(words=["car", "cross", "azdrfa"])
   
   patchscope_probs = patchscope_lens(
       model, source_prompts=source_prompts, target_patch_prompts=target_prompt
   )

Intervene on specific layers (single layer or list):

.. code-block:: python

   # Single layer
   probs = patchscope_lens(
       model, source_prompts=source_prompts, 
       target_patch_prompts=target_prompt, layers=5
   )
   
   # Multiple layers
   probs = patchscope_lens(
       model, source_prompts=source_prompts,
       target_patch_prompts=target_prompt, layers=[3, 5, 7]
   )

Use pre-computed latents instead of source prompts:

.. code-block:: python

   import torch
   
   # Pre-compute activations
   latents = torch.randn(3, 2, 768)  # (num_layers, num_sources, hidden_size)
   
   # Use latents directly (cannot provide source_prompts when using latents)
   probs = patchscope_lens(
       model, latents=latents, 
       target_patch_prompts=target_prompt, layers=[3, 5, 7]
   )

