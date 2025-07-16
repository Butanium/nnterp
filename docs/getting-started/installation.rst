Installation
============

Basic Installation
------------------

.. code-block:: bash

   pip install "nnterp>0.4.9" --pre

For visualization:

.. code-block:: bash

   pip install "nnterp[display]>0.4.9" --pre


Quick Test
----------

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   model = StandardizedTransformer("Maykeye/TinyLLama-v0")
   print("✓ nnterp working!")