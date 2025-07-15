from loguru import logger
from functools import lru_cache
from collections import defaultdict
import copy
import json
import importlib.resources

from tqdm.autonotebook import tqdm
import torch as th
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from huggingface_hub import get_collection
import nnsight
from nnsight import LanguageModel
from nnterp import StandardizedTransformer
from nnterp.utils import dummy_inputs


TRANSFORMERS_VERSION = transformers.__version__
NNSIGHT_VERSION = nnsight.__version__
test_loading_status_path = importlib.resources.files("nnterp.data").joinpath(
    "test_loading_status.json"
)

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

# Core test models
TEST_MODELS = [
    "gpt2",
    "bigscience/bigscience-small-testing",
    "yujiepan/opt-tiny-2layers-random",
    "yujiepan/mixtral-8xtiny-random",
    "yujiepan/qwen1.5-moe-tiny-random",
    "yujiepan/qwen3-moe-tiny-random",
]


def get_all_toy_models():
    return [
        item.item_id
        for item in get_collection(
            "yujiepan/tiny-dummy-models-65acf7ddb68db4f26eb1dec9"
        ).items
    ]


def get_all_test_models(class_names=None) -> list[str]:
    """Get all models used in tests: both collection models and hardcoded ones."""
    collection_models = get_all_toy_models()
    hardcoded_models = TEST_MODELS + LLAMA_LIKE_MODELS
    all_models = list(dict.fromkeys(collection_models + hardcoded_models))
    if class_names is not None:
        all_models = [m for m in all_models if get_arch(m) in class_names]
    return all_models


def load_test_loading_status():
    """Load the test loading status for current transformers/nnsight versions."""
    try:
        with test_loading_status_path.open("r") as f:
            full_status = json.load(f)
    except FileNotFoundError:
        full_status = {}

    if (
        TRANSFORMERS_VERSION not in full_status
        or NNSIGHT_VERSION not in full_status[TRANSFORMERS_VERSION]
    ):
        if TRANSFORMERS_VERSION not in full_status:
            logger.info(
                f"Transformers version {TRANSFORMERS_VERSION} not in status, updating..."
            )
        else:
            logger.info(
                f"NNsight version {NNSIGHT_VERSION} not in status for transformers {TRANSFORMERS_VERSION}, updating..."
            )
        full_status = update_test_loading_status()

    return full_status[TRANSFORMERS_VERSION]


def save_test_loading_status(transformers_status):
    """Save the test loading status dict for current transformers version."""
    try:
        with test_loading_status_path.open("r") as f:
            full_status = json.load(f)
    except FileNotFoundError:
        full_status = {}

    full_status.setdefault(TRANSFORMERS_VERSION, {}).update(transformers_status)

    with open(test_loading_status_path, "w") as f:
        json.dump(full_status, f, indent=4)
    return full_status


def test_model_availability(model_name):
    """Test a single model's HF and nnsight availability. Returns (hf_status, nn_status)."""
    # Test HF availability
    if requires_trust_remote_code(model_name):
        return ("need_trust_remote_code", None)

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=False
        )
    except Exception as e:
        if "trust_remote_code" in str(e):
            return ("need_trust_remote_code", None)
        else:
            return ("cant_load", None)

    try:
        hf_model(th.tensor([[1]]))
    except Exception as e:
        return ("cant_forward", None)

    # Test nnsight availability (only if HF works)
    try:
        nn_model = LanguageModel(model_name)
    except Exception as e:
        try:
            nn_model = LanguageModel(hf_model)
        except Exception as e:
            return ("available_hf", "cant_load_with_hf_in_LanguageModel")
        return ("available_hf", "cant_load_with_LanguageModel")

    try:
        with nn_model.trace(dummy_inputs()):
            pass
    except Exception as e:
        return ("available_hf", "cant_trace_with_LanguageModel")

    return ("available_hf", "available_nn")


def update_test_loading_status():
    """Run full model testing and update the JSON."""
    cant_load = []
    need_trust_remote_code = []
    cant_forward = []
    cant_load_with_LanguageModel = []
    cant_load_with_hf_in_LanguageModel = []
    cant_trace_with_LanguageModel = []
    available_hf_models = []
    available_nn_models = []

    for model_name in tqdm(get_all_test_models()):
        hf_status, nn_status = test_model_availability(model_name)

        if hf_status == "cant_load":
            cant_load.append(model_name)
        elif hf_status == "need_trust_remote_code":
            need_trust_remote_code.append(model_name)
        elif hf_status == "cant_forward":
            cant_forward.append(model_name)
        elif hf_status == "available_hf":
            available_hf_models.append(model_name)

            if nn_status == "cant_load_with_LanguageModel":
                cant_load_with_LanguageModel.append(model_name)
            elif nn_status == "cant_load_with_hf_in_LanguageModel":
                cant_load_with_hf_in_LanguageModel.append(model_name)
            elif nn_status == "cant_trace_with_LanguageModel":
                cant_trace_with_LanguageModel.append(model_name)
            elif nn_status == "available_nn":
                available_nn_models.append(model_name)

    unavailable_hf_models = cant_load + need_trust_remote_code + cant_forward
    unavailable_nn_models = list(
        dict.fromkeys(
            cant_load_with_LanguageModel
            + cant_load_with_hf_in_LanguageModel
            + cant_trace_with_LanguageModel
        )
    )

    transformers_status = {
        "available_hf_models": available_hf_models,
        "unavailable_hf_models": unavailable_hf_models,
        "cant_load_with_AutoModelForCausalLM": cant_load,
        "need_trust_remote_code": need_trust_remote_code,
        "cant_forward": cant_forward,
        NNSIGHT_VERSION: {
            "available_nn_models": available_nn_models,
            "unavailable_nn_models": unavailable_nn_models,
            "cant_load_with_LanguageModel": cant_load_with_LanguageModel,
            "cant_load_with_hf_in_LanguageModel": cant_load_with_hf_in_LanguageModel,
            "cant_trace_with_LanguageModel": cant_trace_with_LanguageModel,
        },
    }

    return save_test_loading_status(transformers_status)


def is_available(model_name, test_status):
    """Check if model is available, using cached status when possible."""
    nnsight_status = test_status.get(NNSIGHT_VERSION, {})
    available_models = nnsight_status.get("available_nn_models", [])
    if model_name in available_models:
        return True, test_status

    unavailable_models = nnsight_status.get(
        "unavailable_nn_models", []
    ) + test_status.get("unavailable_hf_models", [])
    if model_name in unavailable_models:
        return False, test_status

    # Test model if not in cache
    hf_status, nn_status = test_model_availability(model_name)

    # Update HF status in cache
    if hf_status == "cant_load":
        test_status.setdefault("unavailable_hf_models", []).append(model_name)
        test_status.setdefault("cant_load_with_AutoModelForCausalLM", []).append(
            model_name
        )
        return False, test_status
    elif hf_status == "need_trust_remote_code":
        test_status.setdefault("unavailable_hf_models", []).append(model_name)
        test_status.setdefault("need_trust_remote_code", []).append(model_name)
        return False, test_status
    elif hf_status == "cant_forward":
        test_status.setdefault("unavailable_hf_models", []).append(model_name)
        test_status.setdefault("cant_forward", []).append(model_name)
        return False, test_status
    else:
        test_status.setdefault("available_hf_models", []).append(model_name)

    # Update nnsight status in cache
    if nn_status == "cant_load_with_LanguageModel":
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "unavailable_nn_models", []
        ).append(model_name)
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "cant_load_with_LanguageModel", []
        ).append(model_name)
        return False, test_status
    elif nn_status == "cant_load_with_hf_in_LanguageModel":
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "unavailable_nn_models", []
        ).append(model_name)
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "cant_load_with_hf_in_LanguageModel", []
        ).append(model_name)
        return False, test_status
    elif nn_status == "cant_trace_with_LanguageModel":
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "unavailable_nn_models", []
        ).append(model_name)
        test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
            "cant_trace_with_LanguageModel", []
        ).append(model_name)
        return False, test_status

    try:
        StandardizedTransformer(model_name)
    except Exception as e:
        logger.warning(
            f"Model {model_name} can't load with StandardizedTransformer: {e}"
        )
        return False, test_status

    test_status.setdefault(NNSIGHT_VERSION, {}).setdefault(
        "available_nn_models", []
    ).append(model_name)
    return True, test_status


def get_available_models(model_names):
    if not model_names:
        return []
    test_status = load_test_loading_status()
    available_models = []

    for model in model_names:
        available, test_status = is_available(model, test_status)
        if available:
            available_models.append(model)

    save_test_loading_status(test_status)
    return available_models


def get_all_available_models():
    """Get all available models using fast status-based filtering."""
    all_models = list(set(TEST_MODELS + get_all_toy_models() + LLAMA_LIKE_MODELS))
    return get_available_models(all_models)


def get_available_llama_models():
    """Get available models with llama-like naming conventions."""
    return get_available_models(LLAMA_LIKE_MODELS)


def get_failed_models_from_status(test_status):
    """Extract failed models grouped by architecture from test_loading_status.json."""
    failed_models = defaultdict(list)
    nnsight_status = test_status.get(NNSIGHT_VERSION, {})
    unavailable_models = nnsight_status.get("unavailable_nn_models", [])

    for model_name in unavailable_models:
        arch = get_arch(model_name)
        failed_models[arch].append(model_name)

    return failed_models


@lru_cache(maxsize=1000)
def get_model_config_dict(model_name):
    return PretrainedConfig.get_config_dict(model_name)


@lru_cache(maxsize=1000)
def get_arch(model_name):
    archs = get_model_config_dict(model_name)
    if len(archs) == 0:
        return "Unknown"
    archs = archs[0]
    if "architectures" not in archs:
        return "Unknown"
    archs = archs["architectures"]
    if len(archs) == 0:
        return "Unknown"
    if len(archs) == 1:
        return archs[0]
    for _arch in archs:
        if "ForCausalLM" in _arch:
            return _arch
    try:
        AutoModelForCausalLM.from_pretrained(model_name).__class__.__name__
    except Exception as e:
        return "Unknown"


def requires_trust_remote_code(model_name):
    config = get_model_config_dict(model_name)
    if "_auto_map" in config or "auto_map" in config:
        return True
    return False


def merge_partial_status(
    prev_status: dict, new_status: dict, tested_models: defaultdict
) -> dict:
    """Merge partial test results into existing status by updating only the tested models."""
    merged = copy.deepcopy(prev_status)

    for category in new_status:
        for arch in tested_models:
            if category == "nnsight_unavailable_models":
                continue
            if category in merged and arch in merged[category]:
                # Remove tested models from previous categories
                merged[category][arch] = [
                    m for m in merged[category][arch] if m not in tested_models[arch]
                ]
                # Add new models
                merged[category][arch] = list(
                    dict.fromkeys(
                        merged[category][arch] + new_status[category].get(arch, [])
                    )
                )

                if not merged[category][arch]:
                    del merged[category][arch]
            else:

                if category not in merged:
                    merged[category] = {}
                if arch in new_status[category]:
                    merged[category][arch] = new_status[category][arch]

    # Overwrite nnsight_unavailable_models (as it's external)
    if "nnsight_unavailable_models" in new_status:
        merged["nnsight_unavailable_models"] = new_status["nnsight_unavailable_models"]

    return merged


def rm_empty_list(dict_):
    return {k: v for k, v in dict_.items() if v}


if __name__ == "__main__":
    test_status = load_test_loading_status()
    save_test_loading_status(test_status)
