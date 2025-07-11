"""Shared pytest fixtures for all tests."""

import datetime
import json
from collections import defaultdict
from pathlib import Path
import time

import pytest
from _pytest.outcomes import Skipped
from loguru import logger
from nnterp import StandardizedTransformer
from nnsight import LanguageModel

from .utils import (
    get_all_available_models,
    get_arch,
    get_available_llama_models,
    get_failed_models_from_status,
    load_test_loading_status,
    NNSIGHT_VERSION,
    TRANSFORMERS_VERSION,
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def pytest_configure(config):
    """Initialise per-session flags on the config object."""
    config._has_deselected = False
    config._errors = defaultdict(dict)
    config._skips = defaultdict(dict)
    config._fail_test_models = defaultdict(list)
    config._fail_attn_probs_models = defaultdict(list)
    config._tested_models = defaultdict(list)


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
    elif model_name not in config._fail_test_models[arch]:
        logger.warning(f"Model {model_name} failed a test")
        config._fail_test_models[arch].append(model_name)


def pytest_cmdline_main(config):
    config._is_full_run = (
        any(
            Path(p).resolve() == PROJECT_ROOT / "tests"
            for p in config.option.file_or_dir
        )
        or len(config.option.file_or_dir) == 0
    )


def pytest_deselected(items):
    if items:
        items[0].session.config._has_deselected = True


def pytest_sessionfinish(session, exitstatus):
    """Hook called after whole test session finishes."""
    success = exitstatus == 0
    config = session.config
    is_partial = not config._is_full_run or config._has_deselected
    existing_data = {}
    if not is_partial:
        data_file = PROJECT_ROOT / "data" / "status.json"
        data_file.parent.mkdir(exist_ok=True)
        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Error loading {data_file}: {e}")

    try:
        existing_data = _update_status(existing_data, config)
    except Exception as e:
        if existing_data == {}:
            raise e
        logger.warning(f"Error updating status: {e}")
        existing_data = _update_status({}, config)

    if not is_partial:
        with open(data_file, "w") as f:
            json.dump(existing_data, f, indent=2)
        print(
            f"\nModels tested during this session: {config._tested_models}, saving to {data_file}"
        )

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
        "status": existing_data.get(TRANSFORMERS_VERSION, {}).get(NNSIGHT_VERSION, {}),
    }

    with open(log_entry_file, "w") as f:
        json.dump(log_entry, f, indent=2)


def _update_status(prev_status: dict, config):
    nnsight_unavailable_models = get_failed_models_from_status(load_test_loading_status())
    transformers_section = prev_status.setdefault(TRANSFORMERS_VERSION, {})
    new_status = {
        "fully_available_models": {},
        "no_probs_available_models": {},
        "failed_test_models": config._fail_test_models,
        "failed_attn_probs_models": config._fail_attn_probs_models,
        "nnsight_unavailable_models": nnsight_unavailable_models,
        "ran_tests_on": config._tested_models,
    }
    all_failed_tests_models = sum(config._fail_test_models.values(), [])
    all_failed_attn_probs_models = sum(config._fail_attn_probs_models.values(), [])
    all_nnsight_unavailable_models = sum(nnsight_unavailable_models.values(), [])
    all_non_probs_fails = set(all_failed_tests_models + all_nnsight_unavailable_models)
    all_fails = all_non_probs_fails | set(all_failed_attn_probs_models)
    for model_class in config._tested_models:
        fully_available = set(config._tested_models[model_class]) - all_fails
        if fully_available:
            new_status["fully_available_models"][model_class] = list(fully_available)
        available_no_probs = set(config._tested_models[model_class]) - all_non_probs_fails - fully_available
        if available_no_probs:
            new_status["no_probs_available_models"][model_class] = list(available_no_probs)

    new_status["last_updated"] = datetime.datetime.now().isoformat()
    transformers_section[NNSIGHT_VERSION] = new_status
    return prev_status


@pytest.fixture(params=get_all_available_models())
def model_name(request):
    """Parametrized fixture providing test model names."""
    return request.param


@pytest.fixture(params=get_available_llama_models())
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
