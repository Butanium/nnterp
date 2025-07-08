from transformers.configuration_utils import PretrainedConfig
from huggingface_hub import get_collection
from transformers import AutoModelForCausalLM
from tqdm.autonotebook import tqdm
from nnsight import LanguageModel
import json
import transformers
import nnsight
import torch as th


def update_test_loading_status():
    try:
        with open("data/test_loading_status.json", "r") as f:
            test_status = json.load(f)
    except FileNotFoundError:
        test_status = {}

    transformers_version = transformers.__version__
    nnsight_version = nnsight.__version__

    # Initialize structure if needed
    if transformers_version not in test_status:
        test_status[transformers_version] = {}
    if nnsight_version not in test_status[transformers_version]:
        test_status[transformers_version][nnsight_version] = {}
    cant_load = []
    need_trust_remote_code = []
    cant_forward = []
    cant_load_with_LanguageModel = []
    cant_load_with_hf_in_LanguageModel = []
    cant_trace_with_LanguageModel = []
    available_hf_models = []
    available_nn_models = []

    for model_name in tqdm(get_all_toy_models()):
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=False
            )
        except Exception as e:
            if "trust_remote_code" in str(e):
                need_trust_remote_code.append(model_name)
            else:
                cant_load.append(model_name)
            continue
        try:
            hf_model(th.tensor([[1]]))
        except Exception as e:
            cant_forward.append(model_name)
            continue
        available_hf_models.append(model_name)
        try:
            nn_model = LanguageModel(model_name)
        except Exception as e:
            cant_load_with_LanguageModel.append(model_name)
            try:
                nn_model = LanguageModel(hf_model)
            except Exception as e:
                cant_load_with_hf_in_LanguageModel.append(model_name)
                continue
        try:
            with nn_model.trace(th.tensor([[1]])):
                pass
        except Exception as e:
            cant_trace_with_LanguageModel.append(model_name)
            continue
        available_nn_models.append(model_name)

    # Update the status
    test_status[transformers_version]["available_hf_models"] = available_hf_models
    test_status[transformers_version]["cant_load_with_AutoModelForCausalLM"] = cant_load
    test_status[transformers_version]["need_trust_remote_code"] = need_trust_remote_code
    test_status[transformers_version]["cant_forward"] = cant_forward
    test_status[transformers_version][nnsight_version][
        "available_nn_models"
    ] = available_nn_models
    test_status[transformers_version][nnsight_version][
        "cant_load_with_LanguageModel"
    ] = cant_load_with_LanguageModel
    test_status[transformers_version][nnsight_version][
        "cant_load_with_hf_in_LanguageModel"
    ] = cant_load_with_hf_in_LanguageModel
    test_status[transformers_version][nnsight_version][
        "cant_trace_with_LanguageModel"
    ] = cant_trace_with_LanguageModel

    with open("data/test_loading_status.json", "w") as f:
        json.dump(test_status, f, indent=4)


def get_arch(model_name):
    arch = "Unknown"
    archs = PretrainedConfig.get_config_dict(model_name)
    if len(archs) == 0:
        return arch
    archs = archs[0]
    if "architectures" not in archs:
        return arch
    archs = archs["architectures"]
    for _arch in archs:
        if "ForCausalLM" in _arch:
            arch = _arch
            break
    return arch


def get_all_toy_models():
    return [
        item.item_id
        for item in get_collection(
            "yujiepan/tiny-dummy-models-65acf7ddb68db4f26eb1dec9"
        ).items
    ]
