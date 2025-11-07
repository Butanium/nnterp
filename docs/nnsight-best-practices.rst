NNsight Best Practices
======================

.. meta::
   :llm-description: Critical NNsight patterns for nnterp usage. Covers execution order requirements (forward pass order), gradient computation with backward passes, and performance optimization with tracer.stop() and caching.

This guide covers essential NNsight patterns and best practices for efficient and correct usage with nnterp. Understanding these patterns will help you avoid common pitfalls and write more performant code.

Execution Order Requirements
----------------------------

**Critical Rule**: You must access model internals in the same order as the model executes them.

In NNsight, interventions must be written in forward pass order. This means you cannot access layer 2's output before layer 1's output.

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   model = StandardizedTransformer("gpt2")
   
   # ✅ CORRECT: Access layers in forward order
   with model.trace("My tailor is rich"):
       l1 = model.layers_output[1]  # Access layer 1 first
       l2 = model.layers_output[2]  # Then layer 2
       logits = model.logits        # Finally the output
   
   # ❌ INCORRECT: This will fail!
   try:
       with model.trace("My tailor is rich"):
           l2 = model.layers_output[2]  # Access layer 2 first
           l1 = model.layers_output[1]  # Then layer 1 - ERROR!
   except Exception as e:
       print(f"Error: {e}")

This applies to all model components:

.. code-block:: python

   with model.trace("Hello"):
       # ✅ CORRECT: Forward pass order
       attn_0 = model.attentions_output[0]    # Layer 0 attention
       mlp_0 = model.mlps_output[0]           # Layer 0 MLP  
       layer_0 = model.layers_output[0]       # Layer 0 output
       attn_1 = model.attentions_output[1]    # Layer 1 attention
       # ... and so on

Gradient Computation
--------------------

To compute gradients, you must use the ``.backward()`` context and save gradients inside it.

Basic Gradient Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   
   with model.trace("My tailor is rich"):
       # Save the activation you want gradients for
       l1_out = model.layers_output[1]
       
       # Access model output (must be after l1_out)
       logits = model.output.logits
       
       # Compute gradients inside backward context
       with logits.sum().backward():
           l1_grad = l1_out.grad.save()

Multiple Backward Passes
~~~~~~~~~~~~~~~~~~~~~~~~~

For multiple gradient computations, use ``retain_graph=True``:

.. code-block:: python

   with model.trace("My tailor is rich"):
       l1_out = model.layers_output[1]
       logits = model.output.logits
       
       # First backward pass
       with logits.sum().backward(retain_graph=True):
           l1_grad_1 = l1_out.grad.save()
       
       # Second backward pass with different objective
       with (logits.sum() ** 2).backward():
           l1_grad_2 = l1_out.grad.save()
   
   # Gradients will be different
   assert not torch.allclose(l1_grad_1, l1_grad_2)

Don't forget the execution order!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ❌ INCORRECT: Accessing layers_output[1] after logits
   with model.trace("My tailor is rich"):
       logits = model.logits
       with logits.sum().backward():
           # This would fail - can't access layers_output[1] after output
           l1_grad = model.layers_output[1].grad.save()

   # ✅ CORRECT: Save activation first
   with model.trace("My tailor is rich"):
       l1_out = model.layers_output[1]  # Save first
       logits = model.logits
       with logits.sum().backward():
           l1_grad = l1_out.grad.save()  # Use saved activation

Performance Optimization
-------------------------

Use ``tracer.stop()`` to Skip Unnecessary Computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you only need intermediate activations, use ``tracer.stop()`` to prevent the model from computing subsequent layers:

.. code-block:: python

   import time
   
   # Without tracer.stop() - computes all layers
   start = time.time()
   for _ in range(10):
       with model.trace("Hello world"):
           layer_5_out = model.layers_output[5].save()
   time_without_stop = time.time() - start
   
   # With tracer.stop() - only computes up to layer 5
   start = time.time()
   for _ in range(10):
       with model.trace("Hello world") as tracer:
           layer_5_out = model.layers_output[5].save()
           tracer.stop()  # Stop here - don't compute remaining layers
   time_with_stop = time.time() - start
   
   print(f"Speedup: {time_without_stop / time_with_stop:.2f}x")

This can provide significant speedups (often 2-5x) when working with large models and only analyzing intermediate layers.

Caching Activations
-------------------

NNsight 0.5+ includes built-in caching for collecting multiple activations efficiently:

.. code-block:: python

   # Cache activations from multiple layers
   with model.trace("Hello world") as tracer:
       # Cache every other layer
       cache = tracer.cache(
           modules=[model.layers[i] for i in range(0, model.num_layers, 2)]
       ).save()
       
       # Don't call tracer.stop() before cache is accessed!
   
   # Access cached activations
   print(cache.keys())  # Shows module names
   print(cache["model.layers.0"].output.shape)  # Layer 0 output
   print(cache["model.layers.2"].output.shape)  # Layer 2 output

**Important**: As of 0.5.dev8, the cache uses original module names, not nnterp's renamed names.