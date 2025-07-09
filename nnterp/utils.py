import sys
from typing import Union
import torch as th
from nnsight.intervention.tracing.globals import Object

TraceTensor = Union[th.Tensor, Object]


def is_notebook():
    """Detect the current Python environment"""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True
        else:
            return False
    except Exception:
        return False


def display_markdown(text: str):
    from IPython.display import display, Markdown

    display(Markdown(text))


class DummyCache:
    def to_legacy_cache(self):
        return None
