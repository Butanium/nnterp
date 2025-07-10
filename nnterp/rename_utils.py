from enum import Enum
from loguru import logger

import torch as th
from nnsight import Envoy
from .nnsight_utils import TraceTensor
from .utils import is_notebook, display_markdown, try_with_scan


# Dummy class for missing transformer architectures
class ArchitectureNotFound:
    pass


try:
    from transformers import OPTForCausalLM
except ImportError:
    OPTForCausalLM = ArchitectureNotFound

try:
    from transformers import MixtralForCausalLM
except ImportError:
    MixtralForCausalLM = ArchitectureNotFound

try:
    from transformers import BloomForCausalLM
except ImportError:
    BloomForCausalLM = ArchitectureNotFound

try:
    from transformers import GPT2LMHeadModel
except ImportError:
    GPT2LMHeadModel = ArchitectureNotFound

try:
    from transformers import Qwen2MoeForCausalLM
except ImportError:
    Qwen2MoeForCausalLM = ArchitectureNotFound
try:
    from transformers import DbrxForCausalLM
except ImportError:
    DbrxForCausalLM = ArchitectureNotFound


class RenamingError(Exception):
    """Exception raised when the renaming of modules is not properly done."""


# Configuration keys for getting the number of attention heads and hidden size
ATTN_HEAD_CONFIG_KEYS = ["n_heads", "num_attention_heads", "n_head"]
HIDDEN_SIZE_CONFIG_KEYS = ["hidden_size", "d_model", "n_embd"]

# Models that return a tuple for the mlp output
MLP_RETURNS_TUPLE_MODELS = (MixtralForCausalLM, Qwen2MoeForCausalLM, DbrxForCausalLM)

# Alternative names for LLM layers
ATTENTION_NAMES = ["attn", "self_attention", "attention", "norm_attn_norm"]
MODEL_NAMES = ["transformer", "gpt_neox", ".model.decoder"]
LAYER_NAMES = [
    "h",
    "blocks",
    ".decoder.layers",
    ".model.layers",
    ".language_model.layers",
]
LN_NAMES = [
    "final_layer_norm",
    "ln_f",
    "norm_f",
    ".decoder.norm",
    ".model.norm",
    ".language_model.norm",
]
LM_HEAD_NAMES = ["embed_out"]
MLP_NAMES = ["block_sparse_moe", "ffn"]


def text_config(model):
    cfg = model.config
    if "text_config" in cfg:
        cfg = getattr(cfg, "text_config")
    return cfg


def get_num_attention_heads(model):
    cfg = text_config(model)
    for attn_head_key in ATTN_HEAD_CONFIG_KEYS:
        if attn_head_key in cfg:
            return getattr(cfg, attn_head_key)
    raise ValueError(f"No attn head config key found in {model}")


def get_hidden_size(model):
    cfg = text_config(model)
    for hidden_size_key in HIDDEN_SIZE_CONFIG_KEYS:
        if hidden_size_key in cfg:
            return getattr(cfg, hidden_size_key)
    raise ValueError(f"No hidden size config key found in {model}")


def get_rename_dict(
    attn_name: str | list[str] | None = None,
    mlp_name: str | list[str] | None = None,
    ln_final_name: str | list[str] | None = None,
    lm_head_name: str | list[str] | None = None,
    model_name: str | list[str] | None = None,
    layers_name: str | list[str] | None = None,
) -> dict[str, str]:

    rename_dict = (
        {name: "self_attn" for name in ATTENTION_NAMES}
        | {name: "model" for name in MODEL_NAMES}
        | {name: "layers" for name in LAYER_NAMES}
        | {name: "norm" for name in LN_NAMES}
        | {name: "lm_head" for name in LM_HEAD_NAMES}
        | {name: "mlp" for name in MLP_NAMES}
    )

    def update_rename_dict(renaming: str, value: str | list[str] | None):
        if value is not None:
            if isinstance(value, str):
                rename_dict[value] = renaming
            else:
                for name in value:
                    rename_dict[name] = renaming

    update_rename_dict("self_attn", attn_name)
    update_rename_dict("mlp", mlp_name)
    update_rename_dict("norm", ln_final_name)
    update_rename_dict("lm_head", lm_head_name)
    update_rename_dict("model", model_name)
    update_rename_dict("layers", layers_name)

    return rename_dict


class IOType(Enum):
    """Enum to specify input or output access"""

    INPUT = "input"
    OUTPUT = "output"


class LayerAccessor:
    """I/O accessor that provides input/output access with setter"""

    def __init__(
        self,
        model,
        attr_name: str | None,
        io_type: IOType | None,
        returns_tuple: bool = False,
    ):

        self.model = model
        self.attr_name = attr_name
        self.io_type = io_type
        self.returns_tuple = returns_tuple

    def get_module(self, layer: int) -> Envoy:
        module = self.model.layers[layer]
        if self.attr_name is not None:
            module = getattr(module, self.attr_name)
        return module

    def __getitem__(self, layer: int) -> TraceTensor | Envoy:
        module = self.get_module(layer)
        if self.io_type is None:
            return module
        elif self.io_type.value == "input":
            target = module.input
        elif self.io_type.value == "output":
            target = module.output
        else:
            raise ValueError(f"Invalid io_type: {self.io_type}")
        if self.returns_tuple:
            return target[0]
        else:
            return target

    def __setitem__(self, layer: int, value: TraceTensor):
        if self.io_type is None:
            name = self.attr_name or "layers"
            raise ValueError(
                f"Cannot set the value of a module accessor. Did you mean {name}_input/output"
            )
        module = self.get_module(layer)

        if self.io_type.value == "input":
            if self.returns_tuple:
                if len(module.input) > 1:
                    module.input = (value, *module.input[1:])
                else:
                    module.input = (value,)
            else:
                module.input = value
        else:
            if self.returns_tuple:
                if len(module.output) > 1:
                    module.output = (value, *module.output[1:])
                else:
                    module.output = (value,)
            else:
                module.output = value

    def __call__(self, layer: int) -> TraceTensor | Envoy:
        return self[layer]


def bloom_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.self_attention_dropout_0


def default_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.attention_interface_0.source
    else:
        return (
            attention_module.source.attention_interface_0.source.nn_functional_dropout_0
        )


def gpt2_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.attention_interface_0.source
    else:
        return (
            attention_module.source.attention_interface_0.source.module_attn_dropout_0
        )


class AttentionProbabilitiesAccessor:
    def __init__(self, model):
        self.model = model
        if isinstance(model._model, BloomForCausalLM):
            self.source_attr = bloom_attention_prob_source
        elif isinstance(model._model, GPT2LMHeadModel):
            self.source_attr = gpt2_attention_prob_source
        else:
            self.source_attr = default_attention_prob_source
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __getitem__(self, layer: int) -> TraceTensor:
        if not self.enabled:
            raise RenamingError("Attention probabilities are disabled for this model.")
        return self.source_attr(self.model.layers[layer].self_attn).output

    def __setitem__(self, layer: int, value: TraceTensor):
        if not self.enabled:
            raise RenamingError("Attention probabilities are disabled for this model.")
        self.source_attr(self.model.layers[layer].self_attn).output = value

    def check_source(
        self, layer: int = 0, allow_dispatch: bool = True, use_trace: bool = True
    ):

        def test_prob_source():
            batch_size, seq_len = self.model.input_size
            num_heads = get_num_attention_heads(self.model)
            probs = self[layer]
            self[layer] = probs / 2
            if probs.shape != (batch_size, num_heads, seq_len, seq_len):
                raise RenamingError(
                    f"Attention probabilities have shape {probs.shape} != {(batch_size, num_heads, seq_len, seq_len)} (batch_size, n_head, seq_len, seq_len) in {self.model.repo_id} architecture. This means it's not properly initialized."
                )
            if probs.device != th.device("meta"):
                sum_last = probs.sum(dim=-1)
                if not th.allclose(sum_last, th.ones_like(sum_last)):
                    raise RenamingError("Attention probabilities do not sum to 1.")

        if use_trace:
            with self.model.trace(th.tensor([[0, 1, 1]])):
                test_prob_source()
            return

        try_with_scan(
            self.model,
            test_prob_source,
            RenamingError(
                "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
            errors_to_raise=(RenamingError,),
        )

    def print_source(self, layer: int = 0, allow_dispatch: bool = True):
        in_notebook = is_notebook()
        if in_notebook:
            markdown_text = "## Accessing attention probabilities from:\n"
        else:
            print("Accessing attention probabilities from:")

        def print_hook_source():
            nonlocal markdown_text
            source = self.source_attr(self.model.layers[layer].self_attn)
            if in_notebook:
                markdown_text += f"```py\n{source}\n```"
            else:
                print(source)

        used_scan = try_with_scan(
            self.model,
            print_hook_source,
            RenamingError(
                "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
        )
        if in_notebook:
            markdown_text += "\n\n## Full module source:\n"
        else:
            print("\n\nFull module source:")

        def print_attn_source():
            nonlocal markdown_text
            source = str(
                self.source_attr(
                    self.model.layers[layer].self_attn, return_module_source=True
                )
            )
            if in_notebook:
                markdown_text += f"```py\n{source}\n```"
            else:
                print(source)

        try_with_scan(
            self.model,
            print_attn_source,
            RenamingError(
                "Can't access attention probabilities. It is most likely not yet supported for this architecture and transformers version."
            ),
            allow_dispatch=allow_dispatch,
            warn_if_scan_fails=used_scan,
        )

        if in_notebook:
            display_markdown(markdown_text)


IGNORE_MLP_MODELS = (OPTForCausalLM,)


def get_ignores(model) -> list[str]:
    ignores = []
    if isinstance(model, IGNORE_MLP_MODELS):
        message = f"{model.__class__.__name__} does not have a mlp module."
        if isinstance(model, OPTForCausalLM):
            message += " You'll have to manually use layers.fc1 and layers.fc2 instead."
        logger.warning(message)
        ignores.append("mlp")
    return ignores


def mlp_returns_tuple(model) -> bool:
    return isinstance(model, MLP_RETURNS_TUPLE_MODELS)


def check_io(std_model, repo_id: str, ignores: list[str]):
    batch_size, seq_len = std_model.input_size
    hidden_size = get_hidden_size(std_model)
    layer_input = std_model.layers_input[0]
    if not isinstance(layer_input, th.Tensor):
        raise ValueError(
            f"layers_input[0] is not a tensor in {repo_id} architecture. Found type {type(layer_input)}. This means it's not properly initialized."
        )
    if layer_input.shape != (batch_size, seq_len, hidden_size):
        raise ValueError(
            f"layers_input[0] has shape {layer_input.shape} != {(batch_size, seq_len, hidden_size)} in {repo_id} architecture. This means it's not properly initialized."
        )
    if "attention" not in ignores:
        attention_input = std_model.attentions_input[0]
        attention_output = std_model.attentions_output[0]
        if not isinstance(attention_input, th.Tensor):
            raise ValueError(
                f"attentions_input[0] is not a tensor in {repo_id} architecture. Found type {type(attention_input)}. This means it's not properly initialized."
            )
        if attention_input.shape != (batch_size, seq_len, hidden_size):
            raise ValueError(
                f"attentions_input[0] has shape {attention_input.shape} != {(batch_size, seq_len, hidden_size)} in {repo_id} architecture. This means it's not properly initialized."
            )
        if not isinstance(attention_output, th.Tensor):
            raise ValueError(
                f"attentions_output[0] is not a tensor in {repo_id} architecture. Found type {type(attention_output)}. This means it's not properly initialized."
            )
        if attention_output.shape != (
            batch_size,
            seq_len,
            hidden_size,
        ):
            raise ValueError(
                f"attentions_output[0] has shape {attention_output.shape} != {(batch_size, seq_len, hidden_size)} in {repo_id} architecture. This means it's not properly initialized."
            )
    if "mlp" not in ignores:
        mlp_input = std_model.mlps_input[0]
        mlp_output = std_model.mlps_output[0]
        if not isinstance(mlp_input, th.Tensor):
            raise ValueError(
                f"mlps_input[0] is not a tensor in {repo_id} architecture. Found type {type(mlp_input)}. This means it's not properly initialized."
            )
        if mlp_input.shape != (batch_size, seq_len, std_model.config.hidden_size):
            raise ValueError(
                f"mlps_input[0] has shape {mlp_input.shape} != {(batch_size, seq_len, std_model.config.hidden_size)} in {repo_id} architecture. This means it's not properly initialized."
            )
        if not isinstance(mlp_output, th.Tensor):
            raise ValueError(
                f"mlps_output[0] is not a tensor in {repo_id} architecture. Found type {type(mlp_output)}. This means it's not properly initialized."
            )
        if mlp_output.shape != (batch_size, seq_len, std_model.config.hidden_size):
            raise ValueError(
                f"mlps_output[0] has shape {mlp_output.shape} != {(batch_size, seq_len, std_model.config.hidden_size)} in {repo_id} architecture. This means it's not properly initialized."
            )
    layer_output = std_model.layers_output[0]
    if not isinstance(layer_output, th.Tensor):
        raise ValueError(
            f"layers_output[0] is not a tensor in {repo_id} architecture. Found type {type(layer_output)}. This means it's not properly initialized."
        )
    if layer_output.shape != (batch_size, seq_len, std_model.config.hidden_size):
        raise ValueError(
            f"layers_output[0] has shape {layer_output.shape} != {(batch_size, seq_len, std_model.config.hidden_size)} in {repo_id} architecture. This means it's not properly initialized."
        )


def check_model_renaming(
    std_model, repo_id: str, ignores: list[str], allow_dispatch: bool
):

    if not hasattr(std_model, "layers"):
        raise RenamingError(
            f"Could not find layers module in {repo_id} architecture. This means that it was not properly renamed.\n"
            "Please pass the name of the layers module to the layers_rename argument."
        )
    if not hasattr(std_model, "norm"):
        raise RenamingError(
            f"Could not find norm module in {repo_id} architecture. This means that it was not properly renamed.\n"
            "Please pass the name of the norm module to the ln_final_rename argument."
        )
    if not hasattr(std_model, "lm_head"):
        raise RenamingError(
            f"Could not find lm_head module in {repo_id} architecture. This means that it was not properly renamed.\n"
            "Please pass the name of the lm_head module to the lm_head_rename argument."
        )
    if "attention" not in ignores:
        if not hasattr(std_model.layers[0], "self_attn"):
            raise RenamingError(
                f"Could not find self_attn module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the self_attn module to the attn_rename argument."
            )
    if "mlp" not in ignores:
        if not hasattr(std_model.layers[0], "mlp"):
            raise RenamingError(
                f"Could not find mlp module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the mlp module to the mlp_rename argument."
            )

    try_with_scan(
        std_model,
        lambda: check_io(std_model, repo_id, ignores),
        RenamingError(f"Could not check the IO of {repo_id}"),
        allow_dispatch,
        errors_to_raise=(RenamingError,),
    )
