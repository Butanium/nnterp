from __future__ import annotations

import json
import sys
from loguru import logger
from typing import Union
import torch as th
from nnsight.intervention.tracing.globals import Object
import transformers
import nnsight
import importlib.resources
from pathlib import Path
from packaging import version

TraceTensor = Union[th.Tensor, Object]


NNSIGHT_VERSION = nnsight.__version__
TRANSFORMERS_VERSION = transformers.__version__

try:
    status_path = importlib.resources.files("nnterp.data").joinpath("status.json")
    with status_path.open("r") as f:
        STATUS = json.load(f)
except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError) as e:
    logger.warning(f"Error loading status file: {e}")
    STATUS = None


def _get_closest_version(
    current_version: str, available_versions: list[str]
) -> tuple[str | None, str | None]:
    """Get the closest version above and below current version from the available versions"""
    sorted_versions = sorted(available_versions, key=version.parse)
    current_parsed = version.parse(current_version)

    closest_below = None
    closest_above = None

    for v in sorted_versions:
        v_parsed = version.parse(v)
        if v_parsed <= current_parsed:
            closest_below = v
        elif v_parsed >= current_parsed:
            closest_above = v

    return closest_below, closest_above


IS_EXACT_VERSION = True
if STATUS is None:
    nnterp_status = None
else:
    if TRANSFORMERS_VERSION in STATUS:
        transformers_status = STATUS[TRANSFORMERS_VERSION]
    else:
        IS_EXACT_VERSION = False
        available_versions = list(STATUS.keys())
        current_version = TRANSFORMERS_VERSION
        closest_below, closest_above = _get_closest_version(
            current_version, available_versions
        )

        logger.warning(
            f"nnterp was not tested with Transformers version {current_version}. "
            f"Closest below: {closest_below}, closest above: {closest_above}\n"
            f"This is most likely okay, but you may want to at least check that the attention probabilities hook makes sense by calling `model.attention_probabilities.print_source()`. It is recommended to switch to {closest_above or closest_below} if possible or:\n"
            f"  - run the nnterp tests with your version of transformers to ensure everything works as expected.\n  - check if the attention probabilities hook makes sense before using them by calling `model.attention_probabilities.print_source()` (prettier in a notebook).\n"
            f"Using test status from {closest_above or closest_below}."
        )
        transformers_status = STATUS[closest_above or closest_below]

    if NNSIGHT_VERSION in transformers_status:
        nnterp_status = transformers_status[NNSIGHT_VERSION]
    else:
        IS_EXACT_VERSION = False
        compatible_tf_versions = []
        for tf_version, status in STATUS.items():
            if NNSIGHT_VERSION not in status:
                continue
            compatible_tf_versions.append(tf_version)
        tf_closest_below, tf_closest_above = _get_closest_version(
            TRANSFORMERS_VERSION, compatible_tf_versions
        )
        if tf_closest_above:
            action = "upgrade"
        elif tf_closest_below:
            action = "downgrade"
        else:
            action = None
        tf_message = ""
        if action:
            tf_message = f"You could also {action} to transformers {tf_closest_above}, for which NNsight {NNSIGHT_VERSION} was tested."

        available_versions = list(transformers_status.keys())
        current_version = NNSIGHT_VERSION
        closest_below, closest_above = _get_closest_version(
            current_version, available_versions
        )

        logger.warning(
            f"nnterp was not tested with NNsight version {current_version} for transformers version {TRANSFORMERS_VERSION}. "
            f"Closest below: {closest_below}, closest above: {closest_above}\n"
            f"This is most likely okay, but you may want to at least check that the attention probabilities hook makes sense by calling `model.attention_probabilities.print_source()`. It is recommended to switch to NNsight {closest_above or closest_below} if possible.\n"
            + tf_message
            + "Otherwise, consider:\n"
            "  - run the nnterp tests with your version of transformers to ensure everything works as expected using `python -m nnterp run_tests` to update the status file locally.\n"
            "  - check if the attention probabilities hook makes sense before using them by calling `model.attention_probabilities.print_source()` (prettier in a notebook).\n"
            f"Using test status from {closest_above or closest_below}."
        )
        nnterp_status = STATUS[closest_above or closest_below]


if nnterp_status is None:
    CLASS_STATUS = None
else:
    CLASS_STATUS = {
        res_group: list(nnterp_status[res_group].keys())
        for res_group in [
            "fully_available_models",
            "no_probs_available_models",
            "no_intervention_available_models",
            "failed_test_models",
            "failed_attn_probs_models",
            "nnsight_unavailable_models",
        ]
    }


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


def display_source(source: str):
    display_markdown(f"```py\n{source}\n```")


class DummyCache:
    def to_legacy_cache(self):
        return None


def dummy_inputs():
    return {"input_ids": th.tensor([[0, 1, 1]])}


def try_with_scan(
    model,
    function,
    error_to_throw: Exception,
    allow_dispatch: bool,
    warn_if_scan_fails: bool = True,
    errors_to_raise: tuple[type[Exception], ...] | type[Exception] | None = None,
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
        errors_to_raise (tuple, optional): Tuple of exception types that should be raised
            immediately if encountered during scan, without fallback to trace.

    Returns:
        bool: True if scan succeeded, False if trace was used instead
    """

    try:
        with model.scan(dummy_inputs(), use_cache=False) as tracer:
            function()
            tracer.stop()
        return True
    except Exception as e:
        if (
            errors_to_raise
            and errors_to_raise is not None
            and isinstance(e, errors_to_raise)
        ):
            raise e
        if not allow_dispatch and not model.dispatched:
            logger.error("Scan failed and trace() fallback is disabled")
            raise error_to_throw from e
        if warn_if_scan_fails:
            logger.warning(
                "Error when trying to scan the model - using .trace() instead (which will dispatch the model)..."
            )
        try:
            with model.trace(dummy_inputs()) as tracer:
                function()
                tracer.stop()
        except Exception as e2:
            raise error_to_throw from e2
        logger.warning(
            f"Using trace() succeed! Error when trying to scan the model:\n{e}"
        )
        return False


def warn_about_status(class_name: str, model, model_name: str):
    """
    Check the availability of a model for the installed version of NNsight and Transformers.
    Logs a warning if the model is partially or not available.
    """
    if CLASS_STATUS is None:
        return

    if class_name in CLASS_STATUS["fully_available_models"]:
        logger.info(f"{class_name} was tested in nnterp and is fully available.")
        return
    run_tests_str = f"It is advised to run `python -m nnterp run_tests --class-names {class_name}` to run the tests on some toy models with the same architecture. Alternatively, you can run the tests on this model with `python -m nnterp run_tests --model-names {model_name}`."
    if class_name in CLASS_STATUS["failed_test_models"]:
        logger.warning(
            f"The {class_name} class did not pass some basic tests. Use at your own risk. {run_tests_str}"
        )
        return

    if class_name in CLASS_STATUS["nnsight_unavailable_models"]:
        logger.warning(
            f"{class_name} was unavailable in the testing environment and was therefore not tested. {run_tests_str}"
        )
        return

    if (
        class_name in CLASS_STATUS["no_probs_available_models"]
        and model.attn_probs_available()
    ):
        logger.warning(
            f"The {class_name} class did not pass the attention probabilities test,"
            "but still initialized correctly. Please consider using check_attn_probs_with_trace=True to ensure"
            "the attention probabilities pass basic sanity checks. Otherwise, use at your own risk."
        )

    if class_name in CLASS_STATUS["no_prompt_utils_available_models"]:
        logger.warning(
            f"{class_name} failed some tests using `nnterp.prompt_utils`. Use prompt_utils at your own risk. You can also run `python -m nnterp run_tests --class-names {class_name} -k test_prompt_utils` to have more information on the failures."
        )

    if class_name in CLASS_STATUS["no_intervention_available_models"]:
        logger.warning(
            f"{class_name} failed some tests using `nnterp.interventions`. Use interventions at your own risk. You can also run `python -m nnterp run_tests --class-names {class_name} -k test_interventions` to have more information on the failures."
        )
