"""Shared pytest fixtures for all tests."""

import pytest
from nnterp import StandardizedTransformer
from nnsight import LanguageModel
from collections import defaultdict

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


def is_available(model_name):
    if is_available.available[model_name]:
        return True
    elif is_available.available[model_name] is None:
        try:
            model = LanguageModel(model_name)
            with model.trace("a"):
                pass
            is_available.available[model_name] = True
            return True
        except Exception as e:
            is_available.available[model_name] = False
            print(f"Model {model_name} unavailable: {e}")
            return False


is_available.available = defaultdict(lambda: None)


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
