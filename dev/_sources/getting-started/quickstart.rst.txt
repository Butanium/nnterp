Quick Start
===========

This guide gets you up and running with ``nnterp`` in 5 minutes.

Your First Model
---------------

Load any transformer model with a unified interface:

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   # Load any model - they all work the same way
   model = StandardizedTransformer("gpt2")  # or "meta-llama/Llama-2-7b-hf", etc.
   
   # Models use device_map="auto" by default
   print(f"Model loaded on: {model.device}")

Basic Activation Access
----------------------

Access model internals:

.. code-block:: python

   with model.trace("Hello world"):
       # Access layer outputs
       layer_5_output = model.layers_output[5]
       logits = model.logits.save()
   
   # Access attention and MLP components:
   with model.trace("Hello world"):
       attention_output = model.attentions_output[3]
       mlp_output = model.mlps_output[3]

Direct Assignment
-----------------

Set layer outputs directly:

.. code-block:: python

   with model.trace("The capital of France is"):
       # Sum layers 3 and 5, set layer 10 to this value
       combined = model.layers_output[3] + model.layers_output[5] 
       mlp_8 = model.mlps_output[8]  # collect for later
       model.layers_output[10] = combined
       
       # Add MLP output from layer 8 to layer 12 output
       model.layers_output[12] = model.layers_output[12] + mlp_8
       
       result = model.logits.save()

Next Steps
----------

- :doc:`../basic-usage` - Learn the unified interface and module access
- :doc:`../interventions` - Logit lens, patchscope, and steering methods