from transformers.configuration_utils import PretrainedConfig
from huggingface_hub import get_collection


def get_arch(model_name):
    arch = "Unknown"
    archs = PretrainedConfig.get_config_dict(model_name)[0]["architectures"]
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
