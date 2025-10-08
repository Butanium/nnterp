from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
from enum import Enum
from loguru import logger

import torch as th
from nnsight import Envoy
from .nnsight_utils import TraceTensor
from .utils import is_notebook, display_markdown, try_with_scan, dummy_inputs
from .utils import (
    OPTForCausalLM,
    BloomForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    # Additional models from your fork for router functionality
    DbrxForCausalLM,
    GptOssForCausalLM,
    LlamaForCausalLM,
    MixtralForCausalLM,
    OlmoeForCausalLM,
    Qwen2ForCausalLM,
    Qwen2MoeForCausalLM,
    Qwen3ForCausalLM,
)

IgnoreType = Literal["mlp", "attention"]


class RenamingError(Exception):
    """Exception raised when the renaming of modules is not properly done."""


class AttnProbFunction(ABC):
    @abstractmethod
    def get_attention_prob_source(
        self, attention_module, return_module_source: bool = False
    ):
        """
        Get the attention probabilities source for a given attention module. If return_module_source is True,
        return the full module source from where the attention probabilities are computed.
        """
        pass

    def __call__(self, attention_module, return_module_source: bool = False):
        return self.get_attention_prob_source(attention_module, return_module_source)


@dataclass
class RenameConfig:
    """
    Configuration for renaming transformer model modules to standardized names.

    This dataclass specifies how to map model-specific module names to standardized names
    used by nnterp. It allows customization for different transformer architectures.

    Parameters
    ----------
    attn_name : str or list of str, optional
        Name(s) of the attention module to rename to 'self_attn'.

    mlp_name : str or list of str, optional
        Name(s) of the MLP/feed-forward module to rename to 'mlp'.

    ln_final_name : str or list of str, optional
        Name(s) of the final layer normalization to rename to 'ln_final'.

    lm_head_name : str or list of str, optional
        Name(s) of the language model head to rename to 'lm_head'.

    model_name : str or list of str, optional
        Name(s) of the main model container to rename to 'model'.

    layers_name : str or list of str, optional
        Name(s) of the transformer layers container to rename to 'layers'.

    attn_prob_source : AttnProbFunction, optional
        Custom function for accessing attention probabilities.
        Should be an instance of AttnProbFunction that defines how to extract
        attention weights from the attention module.

    ignore_mlp : bool, optional
        Whether to skip MLP module processing for this architecture.
        Some models (e.g., OPT) don't have a unified MLP module.

    ignore_attn : bool, optional
        Whether to skip attention module processing for this architecture.
        Rarely used, for architectures without standard attention.

    attn_head_config_key : str, list of str, or int, optional
        Custom key name for the number of attention heads in model config,
        or the number of heads directly. Defaults to standard keys:
        ['n_heads', 'num_attention_heads', 'n_head'].

    hidden_size_config_key : str, list of str, or int, optional
        Custom key name for hidden size in model config,
        or the hidden size directly. Defaults to standard keys:
        ['hidden_size', 'd_model', 'n_embd'].

    vocab_size_config_key : str, list of str, or int, optional
        Custom key name for vocab size in model config,
        or the vocab size directly. Defaults to standard keys:
        ['vocab_size', 'n_vocab', 'text_config.vocab_size'].

    Example
    -------
    Custom configuration for a non-standard architecture:

        config = RenameConfig(
            attn_name="custom_attention",
            mlp_name=["feed_forward", "ffn"]
        )

    """

    attn_name: str | list[str] | None = None
    mlp_name: str | list[str] | None = None
    ln_final_name: str | list[str] | None = None
    lm_head_name: str | list[str] | None = None
    model_name: str | list[str] | None = None
    layers_name: str | list[str] | None = None
    attn_prob_source: AttnProbFunction | None = None
    ignore_mlp: bool | None = None
    ignore_attn: bool | None = None
    attn_head_config_key: str | list[str] | int | None = None
    hidden_size_config_key: str | list[str] | int | None = None
    vocab_size_config_key: str | list[str] | int | None = None
    router_name: str | list[str] | None = None
    ignore_router: bool | None = None


MODEL_NAMES = ["transformer", "gpt_neox", "decoder", "language_model"]


def expand_path_with_model(paths: list[str]) -> list[str]:
    all_paths = [
        [
            (path.replace("model.", f"model.{model_path}."))
            for path in paths
            if path.startswith("model.")
        ]
        for model_path in MODEL_NAMES
    ]
    return paths + sum(all_paths, [])


# Configuration keys for getting the number of attention heads and hidden size
def default_attn_head_config_keys():
    return ["n_heads", "num_attention_heads", "n_head", "num_heads"]


def default_hidden_size_config_keys():
    return ["hidden_size", "d_model", "n_embd"]


def default_vocab_size_config_keys():
    return ["vocab_size", "n_vocab"]


# Models with no mlp module
IGNORE_MLP_MODELS = (OPTForCausalLM,)
# Models who may squeeze their layer outputs
SQUEEZE_LAYER_OUTPUT_MODELS = (GptOssForCausalLM,)

# Alternative names for LLM layers
ATTENTION_NAMES = ["attn", "self_attention", "attention", "norm_attn_norm"]
LAYER_NAMES = expand_path_with_model(
    [
        "h",
        "blocks",
        "model.layers",
    ]
)
LN_NAMES = expand_path_with_model(
    [
        "final_layer_norm",
        "ln_f",
        "norm_f",
        "norm",
        "embedding_norm",
        "model.ln_final",
    ]
)
LM_HEAD_NAMES = ["embed_out"]
MLP_NAMES = ["block_sparse_moe", "ffn"]
ROUTER_NAMES = ["router", "gate"]
EMBED_TOKENS_NAMES = expand_path_with_model(
    [
        "wte",
        "embed_in",
        "word_embeddings",
        "model.embed_tokens",
    ]
)


def get_rename_dict(
    rename_config: RenameConfig | None = None,
) -> dict[str, str]:
    rename_dict = {}
    if rename_config is not None:

        def update_rename_dict(renaming: str, value: str | list[str] | None):
            if value is not None:
                if isinstance(value, str):
                    rename_dict[value] = renaming
                else:
                    for name in value:
                        rename_dict[name] = renaming

        update_rename_dict("model", rename_config.model_name)
        update_rename_dict("layers", rename_config.layers_name)
        update_rename_dict("self_attn", rename_config.attn_name)
        update_rename_dict("mlp", rename_config.mlp_name)
        update_rename_dict("ln_final", rename_config.ln_final_name)
        update_rename_dict("lm_head", rename_config.lm_head_name)
        update_rename_dict("router", rename_config.router_name)

    rename_dict.update(
        {name: "model" for name in MODEL_NAMES}
        | {name: "layers" for name in LAYER_NAMES}
        | {name: "self_attn" for name in ATTENTION_NAMES}
        | {name: "mlp" for name in MLP_NAMES}
        | {name: "ln_final" for name in LN_NAMES}
        | {name: "lm_head" for name in LM_HEAD_NAMES}
        | {name: "router" for name in ROUTER_NAMES}
        | {name: "embed_tokens" for name in EMBED_TOKENS_NAMES}
    )
    return rename_dict


def text_config(model):
    cfg = model.config
    if "text_config" in cfg:
        cfg = getattr(cfg, "text_config")
    return cfg


def get_num_attention_heads(
    model, raise_error: bool = True, rename_config: RenameConfig | None = None
) -> int | None:
    cfg = text_config(model)
    attn_cfg_keys = default_attn_head_config_keys()
    if rename_config is not None and rename_config.attn_head_config_key is not None:
        if isinstance(rename_config.attn_head_config_key, str):
            attn_cfg_keys.append(rename_config.attn_head_config_key)
        elif isinstance(rename_config.attn_head_config_key, list):
            attn_cfg_keys.extend(rename_config.attn_head_config_key)
        elif isinstance(rename_config.attn_head_config_key, int):
            return rename_config.attn_head_config_key
        else:
            raise ValueError(
                f"Invalid attn head config key: {rename_config.attn_head_config_key}, expected None, str, list[str] or int"
            )
    for attn_head_key in attn_cfg_keys:
        if attn_head_key in cfg:
            return getattr(cfg, attn_head_key)
    if raise_error:
        raise ValueError(f"No attn head config key found in {model}")
    return None


def get_hidden_size(
    model, raise_error: bool = True, rename_config: RenameConfig | None = None
) -> int | None:
    cfg = text_config(model)
    hidden_size_keys = default_hidden_size_config_keys()
    if rename_config is not None and rename_config.hidden_size_config_key is not None:
        if isinstance(rename_config.hidden_size_config_key, str):
            hidden_size_keys.append(rename_config.hidden_size_config_key)
        elif isinstance(rename_config.hidden_size_config_key, list):
            hidden_size_keys.extend(rename_config.hidden_size_config_key)
        elif isinstance(rename_config.hidden_size_config_key, int):
            return rename_config.hidden_size_config_key
        else:
            raise ValueError(
                f"Invalid hidden size config key: {rename_config.hidden_size_config_key}, expected None, str, list[str] or int"
            )
    for hidden_size_key in hidden_size_keys:
        if hidden_size_key in cfg:
            return getattr(cfg, hidden_size_key)
    if raise_error:
        raise ValueError(f"No hidden size config key found in {model}")
    else:
        logger.warning(
            f"Couldn't find the number of attention heads in {model.name_or_path}."
            "You should pass the number of attention heads as an integer or look at the config and pass the key in the attn_head_config_key argument of a RenameConfig."
        )
    return None


def get_vocab_size(
    model, raise_error: bool = True, rename_config: RenameConfig | None = None
) -> int | None:
    cfg = text_config(model)
    vocab_size_keys = default_vocab_size_config_keys()
    if rename_config is not None and rename_config.vocab_size_config_key is not None:
        if isinstance(rename_config.vocab_size_config_key, str):
            vocab_size_keys.append(rename_config.vocab_size_config_key)
        elif isinstance(rename_config.vocab_size_config_key, list):
            vocab_size_keys.extend(rename_config.vocab_size_config_key)
        elif isinstance(rename_config.vocab_size_config_key, int):
            return rename_config.vocab_size_config_key
    for vocab_size_key in vocab_size_keys:
        if vocab_size_key in cfg:
            return getattr(cfg, vocab_size_key)
    if raise_error:
        raise ValueError(f"No vocab size config key found in {model}")
    else:
        return None


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
    ):

        self.model = model
        self.attr_name = attr_name
        self.io_type = io_type
        self._detected_is_tuple: bool | None = None

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

        # Detect tuple status on first access
        if self._detected_is_tuple is None:
            self._detected_is_tuple = isinstance(target, tuple)
        else:
            # Validate consistency
            if isinstance(target, tuple) != self._detected_is_tuple:
                raise RenamingError(
                    f"Inconsistent tuple types detected: layer {layer} has {'tuple' if isinstance(target, tuple) else 'non-tuple'} "
                    f"but expected {'tuple' if self._detected_is_tuple else 'non-tuple'}"
                )

        if self._detected_is_tuple:
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
            if self._detected_is_tuple:
                module.input = (value, *module.input[1:])
            else:
                module.input = value
        else:
            if self._detected_is_tuple:
                module.output = (value, *module.output[1:])
            else:
                module.output = value

    def __call__(self, layer: int) -> TraceTensor | Envoy:
        return self[layer]

    @property
    def returns_tuple(self) -> bool | None:
        """
        Returns whether the layer output is a tuple.
        Returns None if the tuple status has not been detected yet.
        """
        return self._detected_is_tuple


def bloom_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.self_attention_dropout_0


def olmoe_attention_prob_source(attention_module, return_module_source: bool = False):
    """Attention probability source for OlmoeForCausalLM models."""
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.nn_functional_dropout_0


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


def gptj_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source.self__attn_0.source
    else:
        return attention_module.source.self__attn_0.source.self_attn_dropout_0


class AttentionProbabilitiesAccessor:
    def __init__(self, model, rename_config: RenameConfig | None = None):
        self.model = model
        if rename_config is not None and rename_config.attn_prob_source is not None:
            self.source_attr = rename_config.attn_prob_source
        elif isinstance(model._model, BloomForCausalLM):
            self.source_attr = bloom_attention_prob_source
        elif isinstance(model._model, OlmoeForCausalLM):
            self.source_attr = olmoe_attention_prob_source
        elif isinstance(model._model, GPT2LMHeadModel):
            self.source_attr = gpt2_attention_prob_source
        elif isinstance(model._model, GPTJForCausalLM):
            self.source_attr = gptj_attention_prob_source
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
        if self.model.num_heads is None:
            raise RenamingError(
                f"Can't check the shapes of the model internals because the number of attention heads is not available in {self.model.repo_id} architecture."
                "You should pass the number of attention heads as an integer or look at the config and pass the key in the attn_head_config_key argument of a RenameConfig."
            )

        def test_prob_source():
            batch_size, seq_len = self.model.input_size
            num_heads = self.model.num_heads
            probs = self[layer]
            if probs.shape != (batch_size, num_heads, seq_len, seq_len):
                raise RenamingError(
                    f"Attention probabilities have shape {probs.shape} != {(batch_size, num_heads, seq_len, seq_len)} (batch_size, n_head, seq_len, seq_len) in {self.model.repo_id} architecture. This means it's not properly initialized."
                )
            rnd = th.randn_like(probs).abs()
            rnd = rnd / rnd.sum(dim=-1, keepdim=True)
            self[layer] = rnd
            if probs.device != th.device("meta"):
                sum_last = probs.sum(dim=-1)
                if not th.allclose(sum_last, th.ones_like(sum_last)):
                    raise RenamingError("Attention probabilities do not sum to 1.")

        if use_trace:
            with self.model.trace(dummy_inputs()):
                test_prob_source()
                corr_logits = self.model.logits.save()
            with self.model.trace(dummy_inputs()):
                clean_logits = self.model.logits.save()

            if th.allclose(corr_logits, clean_logits):
                raise RenamingError(
                    "Attention probabilities are not properly initialized: changing the attention probabilities should change the logits."
                )
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


def detect_router_attr_name(
    model, layer: int = 0, router_names: list[str] | None = None
) -> str:
    """
    Detect the router attribute name for MoE models.
    
    Args:
        model: The model to inspect
        layer: Layer index to check (default: 0)
        router_names: List of possible router names to check
        
    Returns:
        str: The detected router attribute name
        
    Raises:
        RenamingError: If no router attribute is found
    """
    if router_names is None:
        router_names = ROUTER_NAMES
    
    layer_module = model.model.layers[layer]
    
    # Check if this is an MLP with router
    if hasattr(layer_module, 'mlp'):
        mlp_module = layer_module.mlp
        for router_name in router_names:
            if hasattr(mlp_module, router_name):
                return router_name
    
    # Check if router is directly on the layer
    for router_name in router_names:
        if hasattr(layer_module, router_name):
            return router_name
    
    raise RenamingError(f"Could not find router attribute in layer {layer}. Checked: {router_names}")


def check_router_structure(model, layer: int = 0) -> None:
    """
    Check the router structure for MoE models and log information.
    
    Args:
        model: The model to check
        layer: Layer index to check (default: 0)
    """
    try:
        router_attr = detect_router_attr_name(model, layer)
        layer_module = model.model.layers[layer]
        
        if hasattr(layer_module, 'mlp'):
            router_module = getattr(layer_module.mlp, router_attr)
        else:
            router_module = getattr(layer_module, router_attr)
        
        logger.info(f"Found router '{router_attr}' in layer {layer}")
        logger.info(f"Router module type: {type(router_module)}")
        
        # Check for common router attributes
        if hasattr(router_module, 'weight'):
            logger.info(f"Router weight shape: {router_module.weight.shape}")
        if hasattr(router_module, 'top_k'):
            logger.info(f"Router top_k: {router_module.top_k}")
            
    except RenamingError as e:
        logger.warning(f"Router structure check failed: {e}")


# Router probability computation functions
def compute_router_probabilities(
    router_logits: th.Tensor, top_k: int, norm_topk_prob: bool = True
) -> th.Tensor:
    """
    Convert router logits to probability distribution with top-k selection.

    Implements MoE router probability computation with configurable normalization:
    1. Apply softmax to logits to get initial probabilities
    2. Select top-k experts per token
    3. Set non-selected expert probabilities to 0
    4. Optionally renormalize to ensure probabilities sum to 1

    Args:
        router_logits: Raw router logits of shape [..., num_experts]
        top_k: Number of experts to select per token
        norm_topk_prob: Whether to renormalize probabilities after top-k selection

    Returns:
        Probability distribution of same shape as input
    """
    # Apply softmax to get initial probabilities
    probs = th.softmax(router_logits, dim=-1)

    # Get top-k experts per token
    top_k_probs, top_k_indices = th.topk(probs, k=top_k, dim=-1)

    # Create mask for top-k experts
    mask = th.zeros_like(probs)
    mask.scatter_(-1, top_k_indices, 1.0)

    # Zero out non-selected experts
    masked_probs = probs * mask

    # Optionally renormalize to ensure sum to 1
    if norm_topk_prob:
        prob_sum = masked_probs.sum(dim=-1, keepdim=True)
        masked_probs = masked_probs / prob_sum

    return masked_probs


def compute_default_router_probabilities(
    router_logits: th.Tensor, top_k: int
) -> th.Tensor:
    """
    Compute normalized router probabilities using the default MoE approach.

    This is the standard implementation that normalizes probabilities after top-k selection.
    """
    return compute_router_probabilities(router_logits, top_k, norm_topk_prob=True)


def compute_unnormalized_router_probabilities(
    router_logits: th.Tensor, top_k: int
) -> th.Tensor:
    """
    Compute router probabilities without renormalization after top-k selection.

    Some models don't renormalize after top-k selection.
    """
    return compute_router_probabilities(router_logits, top_k, norm_topk_prob=False)


# Model-specific router probability computation functions
ROUTER_PROBABILITY_FUNCTIONS = {
    OlmoeForCausalLM: compute_unnormalized_router_probabilities,
    # LlamaForCausalLM: compute_sigmoid_router_probabilities,  # future work
}


def get_router_probability_function(model):
    """
    Determine the appropriate router probability function for a given model.

    Args:
        model: The model instance

    Returns:
        Function to compute router probabilities from logits
    """
    # Check for model-specific function
    for model_class, prob_function in ROUTER_PROBABILITY_FUNCTIONS.items():
        if isinstance(model._model, model_class):
            return prob_function

    # Default fallback for unmatched models
    return compute_default_router_probabilities


class RouterProbabilitiesAccessor:
    """
    Specialized accessor for MoE router probability distributions.
    Similar to AttentionProbabilitiesAccessor but for router outputs.

    This accessor computes proper probability distributions from router logits
    using model-specific probability computation functions.
    """

    def __init__(self, model):
        self.model = model
        self.probability_function = get_router_probability_function(model)
        self.enabled = True

    def disable(self):
        """Disable router probabilities access."""
        self.enabled = False

    def _get_router_module(self, layer: int):
        """Get the router module for a specific layer, handling mixed architectures."""
        if not self.enabled:
            raise RenamingError("Router probabilities are disabled for this model.")

        layer_module = self.model.layers[layer]
        if not hasattr(layer_module, "mlp"):
            raise RenamingError(f"Layer {layer} does not have an MLP component.")

        mlp = layer_module.mlp
        if not hasattr(mlp, "router"):
            raise RenamingError(
                f"Layer {layer} does not have a router component ('router')."
            )

        return mlp.router

    def __getitem__(self, layer: int) -> TraceTensor:
        """Get router probability distribution for the specified layer."""
        router = self._get_router_module(layer)
        router_logits = router.output

        # Get top_k for this router
        top_k = self.get_top_k(layer)

        # Compute probabilities using model-specific function
        return self.probability_function(router_logits, top_k)

    def __setitem__(self, layer: int, value: TraceTensor):
        """
        Set router probability distribution for the specified layer.

        This method converts the provided probability distribution back to logits
        and sets the router output, allowing the MoE forward pass to use the
        custom probabilities.

        Args:
            layer: Layer index
            value: Probability distribution tensor with shape (batch_size, seq_len, num_experts)
                  Should be normalized (sum to 1.0 across experts dimension)
        """
        router = self._get_router_module(layer)

        # Validate that value is a proper probability distribution
        assert th.all(value >= 0), "All probabilities must be non-negative"
        prob_sums = value.sum(dim=-1)
        assert th.allclose(prob_sums, th.ones_like(prob_sums), atol=1e-5), (
            "Probabilities must sum to 1 for each token"
        )

        # Convert probabilities back to logits
        # Use -inf for zero probabilities, log for non-zero values
        # Replace zeros with epsilon to avoid log(0) computation
        value_with_epsilon = th.where(value == 0, th.finfo(value.dtype).eps, value)
        logits = th.where(
            value == 0,
            th.tensor(-float("inf"), device=value.device, dtype=value.dtype),
            th.log(value_with_epsilon),
        )

        router.output = logits

    def get_top_k(self, layer: int = 0) -> int:
        """Get the top_k parameter for the router at the specified layer."""
        if not self.enabled:
            raise RenamingError("Router probabilities are disabled for this model.")

        # First try to find a layer with a router
        router = None
        for check_layer in range(layer, len(self.model.layers)):
            try:
                router = self._get_router_module(check_layer)
                break
            except RenamingError:
                continue

        if router is None:
            raise RenamingError(
                f"Could not find any router component starting from layer {layer}"
            )

        # Try different attribute names for top_k
        top_k_attrs = ["top_k", "topk", "num_experts_per_tok", "k"]
        for attr in top_k_attrs:
            if hasattr(router, attr):
                return getattr(router, attr)

        # Check if it's in the model config
        config = self.model.config
        if hasattr(config, "num_experts_per_tok"):
            return config.num_experts_per_tok
        if hasattr(config, "top_k"):
            return config.top_k

        raise RenamingError("Could not find top_k parameter for router")


def get_ignores(model, rename_config: RenameConfig | None = None) -> list[str]:
    ignores = []
    if isinstance(model, IGNORE_MLP_MODELS):
        message = f"{model.__class__.__name__} does not have a mlp module."
        if isinstance(model, OPTForCausalLM):
            message += " You'll have to manually use layers.fc1 and layers.fc2 instead."
        logger.warning(message)
        ignores.append("mlp")
    if rename_config is not None:
        if rename_config.ignore_mlp:
            ignores.append("mlp")
        if rename_config.ignore_attn:
            ignores.append("attention")
        if rename_config.ignore_router:
            ignores.append("router")
    return ignores


def check_io(std_model, model_name: str, ignores: list[IgnoreType]):
    batch_size, seq_len = std_model.input_size
    hidden_size = std_model.hidden_size
    if hidden_size is None:
        raise RenamingError(
            f"Can't check the shapes of the model internals because the hidden size is not available in {model_name} architecture."
            "You should pass the hidden size as an integer or look at the config and pass the key in the hidden_size_config_key argument of a RenameConfig."
        )
    token_embeddings = std_model.token_embeddings
    if not isinstance(token_embeddings, th.Tensor):
        raise ValueError(
            f"token_embeddings is not a tensor in {model_name} architecture. Found type {type(token_embeddings)}. This means it's not properly initialized."
        )
    if token_embeddings.shape != (batch_size, seq_len, hidden_size):
        raise ValueError(
            f"token_embeddings has shape {token_embeddings.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
        )
    layer_input = std_model.layers_input[0]
    if not isinstance(layer_input, th.Tensor):
        raise ValueError(
            f"layers_input[0] is not a tensor in {model_name} architecture. Found type {type(layer_input)}. This means it's not properly initialized."
        )
    if layer_input.shape != (batch_size, seq_len, hidden_size):
        raise ValueError(
            f"layers_input[0] has shape {layer_input.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
        )
    if "attention" not in ignores:
        attention_input = std_model.attentions_input[0]
        attention_output = std_model.attentions_output[0]
        if not isinstance(attention_input, th.Tensor):
            raise ValueError(
                f"attentions_input[0] is not a tensor in {model_name} architecture. Found type {type(attention_input)}. This means it's not properly initialized."
            )
        if attention_input.shape != (batch_size, seq_len, hidden_size):
            raise ValueError(
                f"attentions_input[0] has shape {attention_input.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
            )
        if not isinstance(attention_output, th.Tensor):
            raise ValueError(
                f"attentions_output[0] is not a tensor in {model_name} architecture. Found type {type(attention_output)}. This means it's not properly initialized."
            )
        if attention_output.shape != (
            batch_size,
            seq_len,
            hidden_size,
        ):
            raise ValueError(
                f"attentions_output[0] has shape {attention_output.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
            )
    if "mlp" not in ignores:
        mlp_input = std_model.mlps_input[0]
        mlp_output = std_model.mlps_output[0]
        if not isinstance(mlp_input, th.Tensor):
            raise ValueError(
                f"mlps_input[0] is not a tensor in {model_name} architecture. Found type {type(mlp_input)}. This means it's not properly initialized."
            )
        if mlp_input.shape != (batch_size, seq_len, hidden_size):
            raise ValueError(
                f"mlps_input[0] has shape {mlp_input.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
            )
        if not isinstance(mlp_output, th.Tensor):
            raise ValueError(
                f"mlps_output[0] is not a tensor in {model_name} architecture. Found type {type(mlp_output)}. This means it's not properly initialized."
            )
        if mlp_output.shape != (batch_size, seq_len, hidden_size):
            raise ValueError(
                f"mlps_output[0] has shape {mlp_output.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
            )
    layer_output = std_model.layers_output[0]
    if not isinstance(layer_output, th.Tensor):
        raise ValueError(
            f"layers_output[0] is not a tensor in {model_name} architecture. Found type {type(layer_output)}. This means it's not properly initialized."
        )
    if layer_output.shape != (batch_size, seq_len, hidden_size):
        bad_layer_output_shape_error = ValueError(
            f"layers_output[0] has shape {layer_output.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
        )
        if isinstance(std_model._model, SQUEEZE_LAYER_OUTPUT_MODELS):
            # in this case, it may not be a failure because the model could
            # simply be squeezing a tensor of shape (1, seq_len, hidden_size)
            # into a tensor of shape (seq_len, hidden_size)
            if batch_size != 1:
                raise bad_layer_output_shape_error
            if layer_output.shape != (seq_len, hidden_size):
                raise bad_layer_output_shape_error
        else:
            raise bad_layer_output_shape_error
    ln_final_out = std_model.ln_final.output
    if not isinstance(ln_final_out, th.Tensor):
        raise ValueError(
            f"ln_final.output is not a tensor in {model_name} architecture. Found type {type(ln_final_out)}. This means it's not properly initialized."
        )
    if ln_final_out.shape != (batch_size, seq_len, hidden_size):
        raise ValueError(
            f"ln_final.output has shape {ln_final_out.shape} != {(batch_size, seq_len, hidden_size)} in {model_name} architecture. This means it's not properly initialized."
        )
    lm_head_out = std_model.lm_head.output
    if not isinstance(lm_head_out, th.Tensor):
        raise ValueError(
            f"lm_head.output is not a tensor in {model_name} architecture. Found type {type(lm_head_out)}. This means it's not properly initialized."
        )
    if std_model.vocab_size is None:
        logger.warning(
            f"Couldn't find vocab_size in {model_name} config. Couldn't properly test the shape of lm_head.output."
        )
        if lm_head_out.dim() != 3 or lm_head_out.shape[:-1] != (batch_size, seq_len):
            raise ValueError(
                f"lm_head.output has shape {lm_head_out.shape} != ({batch_size}, { seq_len}, <vocab_size>) in {model_name} architecture. This means it's not properly initialized."
            )
    else:
        if lm_head_out.shape != (batch_size, seq_len, std_model.vocab_size):
            raise ValueError(
                f"lm_head.output has shape {lm_head_out.shape} != {(batch_size, seq_len, std_model.vocab_size)} in {model_name} architecture. This means it's not properly initialized."
            )


def check_model_renaming(
    std_model,
    model_name: str,
    ignores: list[IgnoreType],
    allow_dispatch: bool,
):

    if not hasattr(std_model, "layers"):
        raise RenamingError(
            f"Could not find layers module in {model_name} architecture. This means that it was not properly renamed.\n"
            "Please pass the name of the layers module to the layers_rename argument."
        )
    if not hasattr(std_model, "ln_final"):
        raise RenamingError(
            f"Could not find ln_final module in {model_name} architecture. This means that it was not properly renamed.\n"
            "Please pass the name of the ln_final module to the ln_final_rename argument."
        )
    if not hasattr(std_model, "lm_head"):
        raise RenamingError(
            f"Could not find lm_head module in {model_name} architecture. This means that it was not properly renamed.\n"
            "Please pass the name of the lm_head module to the lm_head_rename argument."
        )
    if "attention" not in ignores:
        if not hasattr(std_model.layers[0], "self_attn"):
            raise RenamingError(
                f"Could not find self_attn module in {model_name} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the self_attn module to the attn_rename argument."
            )
    if "mlp" not in ignores:
        if not hasattr(std_model.layers[0], "mlp"):
            raise RenamingError(
                f"Could not find mlp module in {model_name} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the mlp module to the mlp_rename argument."
            )

    try_with_scan(
        std_model,
        lambda: check_io(std_model, model_name, ignores),
        RenamingError(f"Could not check the IO of {model_name}"),
        allow_dispatch,
        errors_to_raise=(RenamingError,),
    )
