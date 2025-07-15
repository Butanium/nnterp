Installation
============

Basic Installation
------------------

.. code-block:: bash

   pip install nnterp

For visualization:

.. code-block:: bash

   pip install nnterp[display]

Transformers Version
-------------------

``nnterp`` uses ``transformers==4.53.x``. The version is pinned for compatibility with the standardized naming system.

Quick Test
----------

.. code-block:: python

   from nnterp import StandardizedTransformer
   
   model = StandardizedTransformer("Maykeye/TinyLLama-v0")
   print("✓ nnterp working!")