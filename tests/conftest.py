"""Shared pytest fixtures for all tests."""

import datetime
import pytest
import json
from pathlib import Path
from loguru import logger
import transformers
from nnterp import StandardizedTransformer
from nnsight import LanguageModel
from collections import defaultdict
from .utils import get_arch, get_all_toy_models

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


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
    "yujiepan/qwen1.5-tiny-random",
    "yujiepan/qwen2-tiny-random",
    "yujiepan/qwen2.5-tiny-random",
    "yujiepan/qwen3-tiny-random",
    "yujiepan/mistral-nemo-2407-tiny-random",
]

# Centralized model lists
TEST_MODELS = list(
    dict.fromkeys(
        [
            "gpt2",
            "bigscience/bigscience-small-testing",
            "yujiepan/opt-tiny-2layers-random",
            "yujiepan/mixtral-8xtiny-random",
            "yujiepan/qwen1.5-moe-tiny-random",
            "yujiepan/qwen3-moe-tiny-random",
        ]
        + get_all_toy_models()
    )
)

tested_models = defaultdict(list)
failed_models = defaultdict(list)
fail_test_models = defaultdict(list)
fail_attn_probs_models = defaultdict(list)


def pytest_runtest_makereport(item, call):
    if call.when == "call" and call.excinfo is not None:
        model_name = None

        if hasattr(item, "callspec") and hasattr(item.callspec, "params"):
            model_name = item.callspec.params.get("model_name")

        if not model_name and "[" in item.name:
            model_name = item.name.split("[")[1].rstrip("]")

        if model_name:
            arch = get_arch(model_name)
            if "test_probabilities.py" == item.path.name:
                if arch in fail_attn_probs_models:
                    return
                logger.warning(
                    f"Model {model_name} failed the attention probabilities test"
                )
                fail_attn_probs_models[arch].append(model_name)
                return
            if arch in fail_test_models:
                return
            logger.warning(f"Model {model_name} failed a test")
            fail_test_models[arch].append(model_name)
            return


def is_available(model_name):
    if is_available.available[model_name]:
        return True
    elif is_available.available[model_name] is None:
        try:
            model = LanguageModel(model_name)
            with model.trace("a"):
                pass
        except Exception as e:
            is_available.available[model_name] = False

            failed_models[get_arch(model_name)].append(model_name)
            logger.warning(f"Model {model_name} unavailable: {e}")
            return False
        try:
            model = StandardizedTransformer(model_name)
        except Exception as e:
            fail_test_models[get_arch(model_name)].append(model_name)
            logger.warning(
                f"Model {model_name} can't load with StandardizedTransformer: {e}"
            )
            is_available.available[model_name] = False
            return False
        tested_models[model._model.__class__.__name__].append(model_name)
        is_available.available[model_name] = True
        return True


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
    # Only save if all test were run
    if not has_deselected and is_full_run:
        _save_status(exitstatus == 0)


def _update_status(prev_status: dict, success: bool):
    tv = transformers.__version__
    new_status = prev_status.get(tv, {})
    # Initialize status structure
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
        if key not in new_status:
            new_status[key] = default_value

    # Helper function to update model data
    def update_model_data(models_dict, status_models_key, status_classes_key):
        for model_class in models_dict:
            new_status[status_classes_key].append(model_class)
            model_list = new_status[status_models_key].setdefault(model_class, [])
            model_list.extend(models_dict[model_class])
            new_status[status_models_key][model_class] = list(dict.fromkeys(model_list))

    # Update all model data
    update_model_data(tested_models, "tested_models", "tested_classes")
    update_model_data(failed_models, "failed_models", "failed_classes")
    update_model_data(fail_test_models, "failed_test_models", "failed_test_classes")
    update_model_data(
        fail_attn_probs_models, "failed_attn_probs_models", "failed_attn_probs_classes"
    )
    new_status["tested_classes"] = list(
        set(new_status["tested_classes"])
        - set(new_status["failed_classes"])
        - set(new_status["failed_test_classes"])
    )
    new_status["failed_classes"] = list(set(new_status["failed_classes"]))
    new_status["failed_test_classes"] = list(set(new_status["failed_test_classes"]))
    new_status["failed_attn_probs_classes"] = list(
        set(new_status["failed_attn_probs_classes"])
    )
    new_status["last_updated"] = datetime.datetime.now().isoformat()
    new_status["all tests passed"] = success
    prev_status[tv] = new_status
    return prev_status


def _save_status(success: bool):
    """Save the tested models to JSON file."""

    data_file = PROJECT_ROOT / "data" / "status.json"
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
        existing_data = _update_status(existing_data, success)
    except Exception as e:
        logger.warning(f"Error updating status: {e}")
        existing_data = _update_status({}, success)
    # Write back to file
    with open(data_file, "w") as f:
        json.dump(existing_data, f, indent=2)


@pytest.fixture(params=list(set(TEST_MODELS + LLAMA_LIKE_MODELS)))
def model_name(request):
    """Parametrized fixture providing test model names."""
    model_name = request.param
    if not is_available(model_name):
        pytest.skip(f"Model {model_name} unavailable")
    return model_name


@pytest.fixture(params=list(set(LLAMA_LIKE_MODELS)))
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
