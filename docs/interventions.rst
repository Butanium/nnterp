Interventions
=============

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

Patchscope
~~~~~~~~~~

Replace activations from one context into another:

.. code-block:: python

   from nnterp.interventions import patchscope_lens, TargetPrompt, repeat_prompt
   
   source_prompts = ["Paris is beautiful", "London is foggy"]
   target_prompt = TargetPrompt("city: Paris\\nfood: croissant\\n?", -1)
   
   # Or use repeat prompt
   target_prompt = repeat_prompt(words=["car", "cross", "azdrfa"])
   
   patchscope_probs = patchscope_lens(
       model, source_prompts=source_prompts, target_patch_prompts=target_prompt
   )

