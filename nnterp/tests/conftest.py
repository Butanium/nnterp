"""Shared pytest fixtures for all tests."""

import datetime
import json
from collections import defaultdict
import importlib.resources
from pathlib import Path
import time

import pytest
from _pytest.outcomes import Skipped
from loguru import logger
from nnterp import StandardizedTransformer
from nnsight import LanguageModel

from .utils import (
    get_all_available_models,
    get_available_models,
    get_arch,
    get_available_llama_models,
    get_all_test_models,
    get_failed_models_from_status,
    load_test_loading_status,
    NNSIGHT_VERSION,
    TRANSFORMERS_VERSION,
    LLAMA_LIKE_MODELS,
    merge_partial_status,
    rm_empty_list,
    sort_json_recursively,
)

PROJECT_ROOT = Path(str(importlib.resources.files("nnterp")))


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--model-names",
        action="store",
        nargs="+",
        help="If provided, only run tests for the given model names.",
    )
    parser.addoption(
        "--class-names",
        action="store",
        nargs="+",
        help="If provided, only run tests for the given class names.",
    )
    parser.addoption(
        "--save-test-logs",
        action="store_true",
        help="Save test logs to the data/test_logs directory.",
    )


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on command line options."""
    if (
        "model_name" in metafunc.fixturenames
        or "llama_like_model_name" in metafunc.fixturenames
    ):
        model_names = metafunc.config.getoption("model_names", default=None)
        class_names = metafunc.config.getoption("class_names", default=None)
        if model_names is not None or class_names is not None:
            all_models = []
            if model_names is not None:
                all_models = get_available_models(model_names)
            if class_names is not None:
                all_models += get_all_test_models(class_names)

            if not all_models:
                raise ValueError(f"No models available in NNsight from {model_names}")
            llama_like_models = get_available_models(
                [m for m in LLAMA_LIKE_MODELS if m in all_models]
            )
        else:
            all_models = get_all_available_models()
            llama_like_models = get_available_llama_models()

    if "model_name" in metafunc.fixturenames:
        metafunc.parametrize("model_name", all_models)

    if "llama_like_model_name" in metafunc.fixturenames:
        params = (
            llama_like_models
            if llama_like_models
            else [
                pytest.param(
                    None,
                    marks=pytest.mark.skip(reason="No llama-like models available"),
                )
            ]
        )
        metafunc.parametrize("llama_like_model_name", params)


def pytest_configure(config):
    """Initialise per-session flags on the config object."""
    config._has_deselected = False
    config._errors = defaultdict(dict)
    config._skips = defaultdict(dict)
    config._fail_test_models = defaultdict(list)
    config._fail_attn_probs_models = defaultdict(list)
    config._fail_intervention_models = defaultdict(list)
    config._fail_prompt_utils_models = defaultdict(list)
    config._tested_models = defaultdict(list)
    config._is_model_specific = (
        config.getoption("model_names", default=None) is not None
        or config.getoption("class_names", default=None) is not None
    )
    config._is_full_run = (
        any(
            Path(p).resolve() == PROJECT_ROOT / "tests"
            for p in config.option.file_or_dir
        )
        or len(config.option.file_or_dir) == 0
    )
    config._save_test_logs = config.getoption("save_test_logs", default=False)


def pytest_runtest_makereport(item, call):
    if call.when != "call":
        return

    model_name = None

    if hasattr(item, "callspec") and hasattr(item.callspec, "params"):
        model_name = item.callspec.params.get("model_name")

    if not model_name and "[" in item.name:
        model_name = item.name.split("[")[1].rstrip("]")

    if not model_name:
        return
    config = item.session.config
    arch = get_arch(model_name)
    if model_name not in config._tested_models[arch]:
        config._tested_models[arch].append(model_name)

    if call.excinfo is None:
        return

    arch = get_arch(model_name)
    is_skip = call.excinfo.errisinstance(Skipped)
    formatted_tb = str(call.excinfo.getrepr(style="long", chain=True))
    error_message = call.excinfo.exconly()
    if is_skip:
        lst = config._skips[arch].setdefault(model_name, [])
    else:
        lst = config._errors[arch].setdefault(model_name, [])
    lst.append(
        {
            "test_name": item.name,
            "test_file": item.path.name,
            "error": error_message,
            "error_traceback": formatted_tb,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

    if is_skip:
        return
    if item.path.name == "test_probabilities.py":
        if model_name not in config._fail_attn_probs_models[arch]:
            logger.warning(
                f"Model {model_name} failed the attention probabilities test"
            )
            config._fail_attn_probs_models[arch].append(model_name)
    elif item.path.name in ["test_interventions.py", "test_prompt_utils.py"]:
        if model_name not in config._fail_intervention_models[arch]:
            logger.warning(f"Model {model_name} failed the intervention test")
            config._fail_intervention_models[arch].append(model_name)
    elif model_name not in config._fail_test_models[arch]:
        logger.warning(f"Model {model_name} failed a test")
        config._fail_test_models[arch].append(model_name)


def pytest_deselected(items):
    if items:
        items[0].session.config._has_deselected = True


def pytest_sessionfinish(session, exitstatus):
    """Hook called after whole test session finishes."""
    success = exitstatus <= 1
    config = session.config
    if not hasattr(config, "_is_full_run"):
        logger.warning(
            "pytest_sessionfinish called without pytest_configure. Skipping status update."
            "You probably called that with --lf"
        )
        return
    is_partial = not config._is_full_run or config._has_deselected or not success
    existing_data = {}
    if not is_partial:
        status_file = PROJECT_ROOT / "data" / "status.json"
        status_file.parent.mkdir(exist_ok=True)
        if status_file.exists():
            try:
                with open(status_file, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Error loading {status_file}: {e}")

    try:
        new_status = _update_status(existing_data, config)
    except Exception as e:
        if existing_data == {} or config._is_model_specific:
            raise e
        logger.warning(f"Error updating status: {e}")
        new_status = _update_status({}, config)

    if not is_partial:
        with open(status_file, "w") as f:
            json.dump(new_status, f, indent=2)
        print(
            f"\nModels tested during this session: {config._tested_models}, saving to {status_file}"
        )

    if config._save_test_logs:
        log_folder = PROJECT_ROOT / "data" / "test_logs"
        log_folder.mkdir(exist_ok=True)

        file_name = str(int(time.time()))
        if session.config._has_deselected:
            file_name += "_partial"
        log_entry_file = log_folder / f"{file_name}.json"
        print(f"Saving log entry to {log_entry_file}")

        log_entry = {
            "is_full_run": session.config._is_full_run,
            "has_deselected": session.config._has_deselected,
            "transformers_version": TRANSFORMERS_VERSION,
            "nnsight_version": NNSIGHT_VERSION,
            "errors": dict(config._errors),
            "status": new_status.get(TRANSFORMERS_VERSION, {}).get(NNSIGHT_VERSION, {}),
        }

        with open(log_entry_file, "w") as f:
            json.dump(log_entry, f, indent=2)


def _update_status(prev_status: dict, config):
    nnsight_unavailable_models = get_failed_models_from_status(
        load_test_loading_status()
    )
    transformers_section = prev_status.setdefault(TRANSFORMERS_VERSION, {})
    prev_nnsight_status = transformers_section.get(NNSIGHT_VERSION, {})
    config._tested_models = rm_empty_list(config._tested_models)
    config._fail_test_models = rm_empty_list(config._fail_test_models)
    config._fail_attn_probs_models = rm_empty_list(config._fail_attn_probs_models)
    config._fail_intervention_models = rm_empty_list(config._fail_intervention_models)
    config._fail_prompt_utils_models = rm_empty_list(config._fail_prompt_utils_models)
    nnsight_unavailable_models = rm_empty_list(nnsight_unavailable_models)
    new_status = {
        "fully_available_models": {},
        "no_probs_available_models": {},
        "no_intervention_available_models": {},
        "no_prompt_utils_available_models": {},
        "failed_test_models": config._fail_test_models,
        "failed_attn_probs_models": config._fail_attn_probs_models,
        "failed_intervention_models": config._fail_intervention_models,
        "failed_prompt_utils_models": config._fail_prompt_utils_models,
        "nnsight_unavailable_models": nnsight_unavailable_models,
        "ran_tests_on": config._tested_models,
    }
    all_failed_tests_models = sum(config._fail_test_models.values(), [])
    all_failed_attn_probs_models = sum(config._fail_attn_probs_models.values(), [])
    all_failed_intervention_models = sum(config._fail_intervention_models.values(), [])
    all_failed_prompt_utils_models = sum(config._fail_prompt_utils_models.values(), [])
    all_nnsight_unavailable_models = sum(nnsight_unavailable_models.values(), [])
    all_general_fails = set(all_failed_tests_models + all_nnsight_unavailable_models)
    all_fails = (
        all_general_fails
        | set(all_failed_attn_probs_models)
        | set(all_failed_intervention_models)
        | set(all_failed_prompt_utils_models)
    )
    for model_class in config._tested_models:
        fully_available = set(config._tested_models[model_class]) - all_fails
        if len(fully_available) > 0:
            new_status["fully_available_models"][model_class] = sorted(fully_available)
        available_no_probs = (
            set(new_status["failed_attn_probs_models"].get(model_class, []))
            - all_general_fails
        )
        if len(available_no_probs) > 0:
            new_status["no_probs_available_models"][model_class] = sorted(
                available_no_probs
            )
        available_no_intervention = (
            set(new_status["failed_intervention_models"].get(model_class, []))
            - all_general_fails
        )
        if len(available_no_intervention) > 0:
            new_status["no_intervention_available_models"][model_class] = sorted(
                available_no_intervention
            )
        available_no_prompt_utils = (
            set(new_status["failed_prompt_utils_models"].get(model_class, []))
            - all_general_fails
        )
        if len(available_no_prompt_utils) > 0:
            new_status["no_prompt_utils_available_models"][model_class] = sorted(
                available_no_prompt_utils
            )

    if config._is_model_specific:
        new_status = merge_partial_status(
            prev_nnsight_status, new_status, config._tested_models
        )

    new_status["last_updated"] = datetime.datetime.now().isoformat()
    new_status = sort_json_recursively(new_status, preserve_level=0)
    transformers_section[NNSIGHT_VERSION] = new_status
    return prev_status


@pytest.fixture
def model_name(request):
    """Parametrized fixture providing test model names."""
    return request.param


@pytest.fixture
def llama_like_model_name(request):
    """Parametrized fixture for models with llama-like naming conventions."""
    return request.param


@pytest.fixture
def model(model_name):
    """Fixture providing StandardizedTransformer instances."""
    return StandardizedTransformer(model_name)


@pytest.fixture
def raw_model(model_name):
    """Fixture providing raw LanguageModel instances."""
    return LanguageModel(model_name, device_map="auto")
