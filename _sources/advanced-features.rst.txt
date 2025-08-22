Advanced Features
=================

Attention Probabilities
-----------------------

For supported models, access attention probabilities:

.. code-block:: python

   with model.trace("The cat sat on the mat"):
       # Access attention probabilities for layer 5
       attn_probs = model.attention_probabilities[5].save()
       # Shape: (batch, heads, seq_len, seq_len)
       
       # Modify attention patterns
       attn_probs[:, :, :, 0] = 0  # Remove attention to first token
       attn_probs /= attn_probs.sum(dim=-1, keepdim=True)  # Renormalize
       
       modified_logits = model.logits.save()

Check what's happening:

.. code-block:: python

   model.attention_probabilities.print_source()

Prompt Utilities
----------------

Track probabilities of specific tokens:

.. code-block:: python

   from nnterp.prompt_utils import Prompt, run_prompts
   
   # Create prompt with target tokens
   prompt = Prompt.from_strings(
       "The capital of France (not England or Spain) is",
       {
           "target": "Paris",
           "traps": ["London", "Madrid"],
           "longstring": "the country of France",
       },
       model.tokenizer,
   )
   
   # Check what tokens are tracked
   for name, tokens in prompt.target_tokens.items():
       print(f"{name}: {model.tokenizer.convert_ids_to_tokens(tokens)}")
   
   # Get probabilities
   results = run_prompts(model, [prompt])
   for target, probs in results.items():
       print(f"{target}: {probs.shape}")  # Shape: (batch_size,)

Combined with Interventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nnterp.interventions import logit_lens
   
   # Use interventions with target tracking
   results = run_prompts(model, prompts, get_probs_func=logit_lens)
   # Returns probabilities for each target category across all layers

Visualization
-------------

Plot top tokens at each layer:

.. code-block:: python

   from nnterp.display import plot_topk_tokens, prompts_to_df
   
   probs = logit_lens(model, "The capital of France is")
   
   # Interactive plot
   plot_topk_tokens(
       probs[0],  # First prompt
       model.tokenizer,
       k=5,
       title="Top 5 tokens at each layer"
   )

Prompt Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert prompts to DataFrame
   df = prompts_to_df(prompts, model.tokenizer)
   display(df)

Plot Target Evolution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import plotly.graph_objects as go
   
   # From logit lens results with target tracking
   results = run_prompts(model, prompts, get_probs_func=logit_lens)
   
   fig = go.Figure()
   for category, probs in results.items():
       fig.add_trace(go.Scatter(
           x=list(range(len(probs[0]))),
           y=probs[0].tolist(),
           mode="lines+markers",
           name=category
       ))
   
   fig.update_layout(
       title="Target Token Probabilities Across Layers",
       xaxis_title="Layer", 
       yaxis_title="Probability"
   )
   fig.show()