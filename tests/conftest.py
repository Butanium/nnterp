"""Shared pytest fixtures for all tests."""

import datetime
import json
from collections import defaultdict
from pathlib import Path
import time

import pytest
import transformers
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
    config._errors[arch].setdefault(model_name, []).append(
        {
            "test_name": item.name,
            "test_file": item.path.name,
            "skipped": is_skip,
            "error": error_message,
            "error_traceback": formatted_tb,
            "timestamp": datetime.datetime.now().isoformat(),
            "call_excinfo": str(call.excinfo),
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
        existing_data = _update_status(existing_data, success, config)
    except Exception as e:
        if existing_data == {}:
            raise e
        logger.warning(f"Error updating status: {e}")
        existing_data = _update_status({}, success, config)

    if not is_partial:
        with open(data_file, "w") as f:
            json.dump(existing_data, f, indent=2)
        print(
            f"\nModels tested during this session: {config._tested_models}, saving to {data_file}"
        )

    log_folder = PROJECT_ROOT / "data" / "test_logs"
    log_folder.mkdir(exist_ok=True)

    timestamp = int(time.time())
    log_entry_file = log_folder / f"{timestamp}.json"
    print(f"Saving log entry to {log_entry_file}")

    log_entry = {
        "is_full_run": session.config._is_full_run,
        "has_deselected": session.config._has_deselected,
        "transformers_version": TRANSFORMERS_VERSION,
        "nnsight_version": NNSIGHT_VERSION,
        "errors": dict(config._errors),
        "status": existing_data[TRANSFORMERS_VERSION],
    }

    with open(log_entry_file, "w") as f:
        json.dump(log_entry, f, indent=2)


def _update_status(prev_status: dict, success: bool, config):
    new_status = prev_status.get(TRANSFORMERS_VERSION, {})

    defaults = {
        "tested_models": {},
        "tested_classes": [],
        "failed_models": {},
        "failed_classes": [],
        "failed_test_models": {},
        "failed_test_classes": [],
        "failed_attn_probs_models": {},
        "failed_attn_probs_classes": [],
    }
    for key, default_value in defaults.items():
        new_status.setdefault(key, default_value)

    def update_model_data(models_dict, status_models_key, status_classes_key):
        for model_class in models_dict:
            new_status[status_classes_key].append(model_class)
            model_list = new_status[status_models_key].setdefault(model_class, [])
            model_list.extend(models_dict[model_class])
            new_status[status_models_key][model_class] = list(dict.fromkeys(model_list))

    test_loading_status = load_test_loading_status()

    failed_models = get_failed_models_from_status(test_loading_status)

    update_model_data(config._tested_models, "tested_models", "tested_classes")
    update_model_data(failed_models, "failed_models", "failed_classes")
    update_model_data(
        config._fail_test_models, "failed_test_models", "failed_test_classes"
    )
    update_model_data(
        config._fail_attn_probs_models,
        "failed_attn_probs_models",
        "failed_attn_probs_classes",
    )
    new_status["all_tests_passed"] = success
    new_status["tested_classes"] = sorted(
        set(new_status["tested_classes"])
        - set(new_status["failed_classes"])
        - set(new_status["failed_test_classes"])
    )
    new_status["failed_classes"] = sorted(set(new_status["failed_classes"]))
    new_status["failed_test_classes"] = sorted(set(new_status["failed_test_classes"]))
    new_status["failed_attn_probs_classes"] = sorted(
        set(new_status["failed_attn_probs_classes"])
    )

    # Incorporate test loading status data if available
    if TRANSFORMERS_VERSION in test_loading_status:
        new_status["loading_status"] = test_loading_status[TRANSFORMERS_VERSION]

    new_status["last_updated"] = datetime.datetime.now().isoformat()
    prev_status[TRANSFORMERS_VERSION] = new_status
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
