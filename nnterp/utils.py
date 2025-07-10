import sys
from loguru import logger
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


def try_with_scan(
    model,
    function,
    error_to_throw: Exception,
    allow_dispatch: bool,
    warn_if_scan_fails: bool = True,
):
    """
    Attempt to execute a function using model.scan(), falling back to model.trace() if needed.

    This function tries to execute the given function within a model.scan() context first,
    which avoids dispatching the model. If that fails and fallback is allowed, it will
    try using model.trace() instead, which does dispatch the model.

    Args:
        model: The model object that supports .scan() and .trace() methods
        function: A callable to execute within the model context (takes no arguments)
        error_to_throw (Exception): Exception to raise if both scan and trace fail
        allow_dispatch (bool): Whether to allow fallback to .trace() if .scan() fails
        warn_if_scan_fails (bool, optional): Whether to log warnings when scan fails.
            Defaults to True.

    Returns:
        bool: True if scan succeeded, False if trace was used instead
    """

    try:
        with model.scan("a", use_cache=False):
            function()
        return True
    except Exception as e:
        if not allow_dispatch:
            logger.error("Scan failed and trace() fallback is disabled")
            raise error_to_throw from e
        if warn_if_scan_fails:
            logger.warning(
                "Error when trying to scan the model - using .trace() instead (which will dispatch the model)..."
            )
        try:
            with model.trace("a"):
                function()
        except Exception as e2:
            raise error_to_throw from e2
        logger.warning(
            f"Using trace() succeed! Error when trying to scan the model:\n{e}"
        )
        return False
