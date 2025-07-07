"""Shared pytest fixtures for all tests."""

import datetime
import pytest
import json
from pathlib import Path
from loguru import logger
import transformers
from transformers import AutoModelForCausalLM
from nnterp import StandardizedTransformer
from nnsight import LanguageModel
from collections import defaultdict
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Centralized model lists
TEST_MODELS = [
    "gpt2",
    "bigscience/bigscience-small-testing",
    "yujiepan/opt-tiny-2layers-random",
    "yujiepan/mixtral-8xtiny-random",
]

# Models with llama-like naming conventions
LLAMA_LIKE_MODELS = [
    "sbintuitions/tiny-lm-chat",
    "Maykeye/TinyLLama-v0",
    "axolotl-ai-co/gemma-3-34M",
    "yujiepan/gemma-tiny-random",
    "yujiepan/llama-4-tiny-random",
    "yujiepan/llama-3.3-tiny-random",
    "yujiepan/llama-2-tiny-random",
    "yujiepan/mistral-tiny-random",
    "yujiepan/deepseek-llm-tiny-random",
    "yujiepan/phi-3-tiny-random",
    "yujiepan/phi-3.5-moe-tiny-random",
]

tested_models = defaultdict(list)
failed_models = defaultdict(list)


def is_available(model_name):
    if is_available.available[model_name]:
        return True
    elif is_available.available[model_name] is None:
        try:
            model = LanguageModel(model_name)
            with model.trace("a"):
                pass
            tested_models[model._model.__class__.__name__].append(model_name)
            is_available.available[model_name] = True
            return True
        except Exception as e:
            is_available.available[model_name] = False
            failed_models[
                AutoModelForCausalLM.from_pretrained(model_name).__class__.__name__
            ].append(model_name)
            logger.warning(f"Model {model_name} unavailable: {e}")
            return False


is_available.available = defaultdict(lambda: None)
is_full_run = False
has_deselected = False


def pytest_cmdline_main(config):
    for file_or_dir in config.option.file_or_dir:
        if Path(file_or_dir).resolve() == PROJECT_ROOT / "tests":
            global is_full_run
            is_full_run = True


def pytest_deselected(items):
    global has_deselected
    has_deselected = True


def pytest_sessionfinish(session, exitstatus):
    """Hook called after whole test session finishes."""
    # Only save if all tests passed/skipped/xfailed (no failures) and no tests were deselected
    if exitstatus == 0 and not has_deselected and is_full_run:
        _save_status(exitstatus == 0)


def _update_status(prev_status: dict):
    tv = transformers.__version__
    new_status = prev_status.get(tv, {})
    if "tested_models" not in new_status:
        new_status["tested_models"] = {}
    if "failed_models" not in new_status:
        new_status["failed_models"] = {}
    if "tested_classes" not in new_status:
        new_status["tested_classes"] = []
    if "failed_classes" not in new_status:
        new_status["failed_classes"] = []

    for model_class in tested_models:
        new_status["tested_classes"].append(model_class)
        tested = new_status["tested_models"].get(model_class, [])
        tested.extend(tested_models[model_class])
        new_status["tested_models"][model_class] = list(dict.fromkeys(tested))
    for model_class in failed_models:
        new_status["failed_classes"].append(model_class)
        failed = new_status["failed_models"].get(model_class, [])
        failed.extend(failed_models[model_class])
        new_status["failed_models"][model_class] = list(dict.fromkeys(failed))
    new_status["tested_classes"] = list(
        set(new_status["tested_classes"]) - set(new_status["failed_classes"])
    )
    new_status["failed_classes"] = list(set(new_status["failed_classes"]))
    new_status["last_updated"] = datetime.datetime.now().isoformat()
    new_status["transformers_versions"] = list(
        dict.fromkeys(new_status["transformers_versions"] + [tv])
    )
    prev_status[tv] = new_status
    return prev_status


def _save_status(success: bool):
    """Save the tested models to JSON file."""

    data_file = (
        PROJECT_ROOT / "data" / "status.json"
        if success
        else PROJECT_ROOT / "data" / "status_failed.json"
    )
    print(
        f"\nModels tested during this session: {tested_models}, saving to {data_file}"
    )
    data_file.parent.mkdir(exist_ok=True)  # Ensure data directory exists

    existing_data = {}
    if data_file.exists():
        try:
            with open(data_file, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Error loading {data_file}: {e}")

    try:
        existing_data = _update_status(existing_data)
    except Exception as e:
        logger.warning(f"Error updating status: {e}")
        existing_data = _update_status({})

    # Write back to file
    with open(data_file, "w") as f:
        json.dump(existing_data, f, indent=2)


def update_models():
    for model_name in tqdm(TEST_MODELS + LLAMA_LIKE_MODELS):
        is_available(model_name)
    _save_status(True)


@pytest.fixture(params=TEST_MODELS + LLAMA_LIKE_MODELS)
def model_name(request):
    """Parametrized fixture providing test model names."""
    model_name = request.param
    if not is_available(model_name):
        pytest.skip(f"Model {model_name} unavailable")
    return model_name


@pytest.fixture(params=LLAMA_LIKE_MODELS)
def llama_like_model_name(request):
    """Parametrized fixture for models with llama-like naming conventions."""
    model_name = request.param
    if not is_available(model_name):
        pytest.skip(f"Model {model_name} unavailable")
    return model_name


@pytest.fixture
def model(model_name):
    """Fixture providing StandardizedTransformer instances."""
    return StandardizedTransformer(model_name)


@pytest.fixture
def raw_model(model_name):
    """Fixture providing raw LanguageModel instances."""
    return LanguageModel(model_name, device_map="auto")
