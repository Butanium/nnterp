from __future__ import annotations
import json
from collections import defaultdict
from warnings import warn
from pathlib import Path
import torch as th
from nnsight import LanguageModel
from loguru import logger
from nnsight.intervention.envoy import Envoy
import transformers

from .nnsight_utils import (
    TraceTensor,
    get_num_layers,
    get_logits,
    get_unembed_norm,
    project_on_vocab,
    skip_layer,
    skip_layers,
    get_next_token_probs,
    GetModuleOutput,
    get_layer_output,
)
from .rename_utils import (
    get_rename_dict,
    LayerAccessor,
    IOType,
    get_ignores,
    mlp_returns_tuple,
    check_model_renaming,
    AttentionProbabilitiesAccessor,
)


status_file = Path("data/status.json")
if status_file.exists():
    with open(status_file, "r") as f:
        status = json.load(f)
    if transformers.__version__ not in status:
        # Find the closest versions above and below current transformers version
        available_versions = list(status.keys())
        current_version = transformers.__version__

        # Sort versions to find closest matches
        from packaging import version

        sorted_versions = sorted(available_versions, key=version.parse)
        current_parsed = version.parse(current_version)

        closest_below = None
        closest_above = None

        for v in sorted_versions:
            v_parsed = version.parse(v)
            if v_parsed <= current_parsed:
                closest_below = v
            elif v_parsed > current_parsed and closest_above is None:
                closest_above = v

        logger.warning(
            f"nnterp was not tested with Transformers version {current_version}. "
            f"Closest below: {closest_below}, closest above: {closest_above}\n"
            f"This is most likely okay, but you may want to at least check that the attention probabilities hook makes sense by calling `model.attention_probabilities.print_source()`. It is recommended to switch to {closest_above} or {closest_below} if possible or:\n"
            f"  - run the nnterp tests with your version of transformers to ensure everything works as expected.\n  - check if the attention probabilities hook makes sense before using them by calling `model.attention_probabilities.print_source()` (prettier in a notebook)."
        )
        status = status[closest_above or closest_below]
    else:
        status = status[transformers.__version__]
        status = defaultdict(list, status)

else:
    logger.warning(f"Status file {status_file} not found. Can't access tested models.")
    status = None


def load_model(
    model_name: str,
    trust_remote_code=False,
    no_space_on_bos=False,
    rename_kwargs=None,
    use_tl=False,
    **kwargs_,
):
    """
    Load a model into nnsight. If use_tl is True, a TransformerLens model is loaded.
    Default device is "auto" and default torch_dtype is th.float16.

    Args:
        no_space_on_bos: If True, add_prefix_space is set to False in the tokenizer. It is useful if you want to use the tokenizer to get the first token of a word when it's not after a space.
    """
    warn(
        "This function is deprecated and will be removed in the future. Use nnterp.StandardizedTransformer instead."
    )
    if rename_kwargs is None:
        rename_kwargs = {}
    kwargs = dict(torch_dtype=th.bfloat16, trust_remote_code=trust_remote_code)
    if use_tl:
        raise ValueError("TransformerLens is no longer supported as of nnterp v1.0.0")
    else:
        kwargs["device_map"] = "auto"
        tokenizer_kwargs = kwargs_.pop("tokenizer_kwargs", {})
        if no_space_on_bos:
            tokenizer_kwargs.update(
                dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
            )
        kwargs.update(kwargs_)
        rename_modules_dict = get_rename_dict(**rename_kwargs)
        model = StandardizedTransformer(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
            rename=rename_modules_dict,
            **kwargs,
        )
        return model


class StandardizedTransformer(LanguageModel):
    """
    Renames the LanguageModel modules to match a standardized architecture.

    The model structure is organized as follows::

        StandardizedTransformer
            ├── model
            │   ├── layers
            │   │   ├── self_attn
            │   │   └── mlp
            │   └── norm
            └── lm_head

    In addition to renaming modules, this class provides built-in accessors to extract and set intermediate activations:

    - layers[i]: Get layer module at layer i
    - layers_input[i]: Get/set layer input at layer i
    - layers_output[i]: Get/set layer output at layer i
    - attentions_output[i]: Get/set attention output at layer i
    - attentions[i]: Get attention module at layer i
    - mlps_output[i]: Get/set MLP output at layer i
    - mlps[i]: Get MLP module at layer i

    Args:
        repo_id (str): Hugging Face repository ID or path of the model to load.
        trust_remote_code (bool, optional): If True, remote code will be trusted when
            loading the model. Defaults to False.
        attn_rename (str, optional): Extra module name to rename to ``self_attn``.
        mlp_rename (str, optional): Extra module name to rename to ``mlp``.
        ln_final_rename (str, optional): Extra module name to rename to ``ln_final``.
        lm_head_rename (str, optional): Extra module name to rename to ``lm_head``.
        model_rename (str, optional): Extra module name to rename to ``model``.
        layers_rename (str, optional): Extra module name to rename to ``layers``.
        check_renaming (bool, optional): If True, the renaming of modules is validated.
            Defaults to True.
    """

    def __init__(
        self,
        repo_id: str,
        trust_remote_code: bool = False,
        attn_rename: str | None = None,
        mlp_rename: str | None = None,
        ln_final_rename: str | None = None,
        lm_head_rename: str | None = None,
        model_rename: str | None = None,
        layers_rename: str | None = None,
        check_renaming: bool = True,
        fallback_check_to_trace: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("device_map", "auto")
        # Check if attention implementation is supported for attention pattern tracing
        if "attn_implementation" in kwargs or "config" in kwargs:
            impl = (
                kwargs.pop("attn_implementation", None)
                or kwargs["config"]._attn_implementation
            )
            if impl != "eager":
                warn(
                    f"Attention implementation {impl} is not supported for attention pattern tracing. Please use eager attention implementation if you plan to access attention patterns."
                )
        else:
            logger.info(
                "Enforcing eager attention implementation for attention pattern tracing. The HF default would be to use sdpa if available. To use sdpa, set attn_implementation='sdpa' or None to use the HF default."
            )
            impl = "eager"
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        rename = get_rename_dict(
            attn_name=attn_rename,
            mlp_name=mlp_rename,
            ln_final_name=ln_final_rename,
            lm_head_name=lm_head_rename,
            model_name=model_rename,
            layers_name=layers_rename,
        )
        user_rename = kwargs.pop("rename", None)
        if user_rename is not None:
            logger.info(
                f"Updating default rename with user-provided rename: {user_rename}"
            )
            rename.update(user_rename)
        super().__init__(
            repo_id,
            attn_implementation=impl,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            rename=rename,
            **kwargs,
        )
        if status is not None:
            if self._model.__class__.__name__ in status["failed_test_classes"]:
                logger.warning(
                    f"{repo_id}'s architecture has failed tests for this transformer version. Use at your own risks. If you want to be safe use only the renaming feature of nnterp, and do not use model.layers_output and other accessors"
                )
            if self._model.__class__.__name__ not in status["tested_classes"]:
                logger.warning(
                    f"{repo_id}'s architecture is not tested. This may cause unexpected behavior. It is recommended to check that the attention probabilities hook makes sense by calling `model.attention_probabilities.print_source()` if you plan on using it (prettier in a notebook).\nFeel free to open an issue on github (https://github.com/butanium/nnterp/issues) or run the tests yourself with a toy model if you want to add test coverage for this model."
                )
            elif self._model.__class__.__name__ in status["failed_attn_probs_classes"]:
                logger.warning(
                    f"{repo_id}'s architecture has failed attention probabilities tests for this transformer version. Do not use model.attention_probabilities"
                )
        ignores = get_ignores(self._model)

        # Create accessor instances
        self.layers_input = LayerAccessor(self, None, IOType.INPUT, returns_tuple=False)
        self.layers_output = LayerAccessor(
            self, None, IOType.OUTPUT, returns_tuple=True
        )
        self.attentions = LayerAccessor(self, "self_attn", None)
        self.attentions_input = LayerAccessor(
            self, "self_attn", IOType.INPUT, returns_tuple=False
        )
        self.attentions_output = LayerAccessor(
            self, "self_attn", IOType.OUTPUT, returns_tuple=True
        )
        self.mlps = LayerAccessor(self, "mlp", None)
        self.mlps_input = LayerAccessor(self, "mlp", IOType.INPUT, returns_tuple=False)
        self.mlps_output = LayerAccessor(
            self,
            "mlp",
            IOType.OUTPUT,
            returns_tuple=mlp_returns_tuple(self._model),
        )
        if check_renaming:
            check_model_renaming(self, repo_id, ignores, fallback_check_to_trace)
        self.num_layers = get_num_layers(self)
        self.attention_probabilities = AttentionProbabilitiesAccessor(self)

    @property
    def unembed_norm(self) -> Envoy:
        return get_unembed_norm(self)

    def project_on_vocab(self, h: TraceTensor) -> TraceTensor:
        return project_on_vocab(self, h)

    @property
    def logits(self) -> TraceTensor:
        """Returns the lm_head output"""
        return get_logits(self)

    @property
    def next_token_probs(self) -> TraceTensor:
        return get_next_token_probs(self)

    def skip_layer(self, layer: int, skip_with: TraceTensor | None = None):
        """
        Skip the computation of a layer.

        Args:
            layer: The layer to skip
            skip_with: The input to skip the layer with. If None, the input of the layer is used.
        """
        return skip_layer(self, layer, skip_with)

    def skip_layers(
        self, start_layer: int, end_layer: int, skip_with: TraceTensor | None = None
    ):
        """
        Skip all layers between start_layer and end_layer (inclusive).

        Args:
            start_layer: The layer to start skipping from
            end_layer: The layer to stop skipping at (inclusive)
            skip_with: The input to skip the layers with. If None, the input of start_layer is used.
        """
        return skip_layers(self, start_layer, end_layer, skip_with)

    def steer(
        self,
        layers: int | list[int],
        steering_vector: th.Tensor,
        factor: float = 1,
        positions: int | list[int] | th.Tensor | None = None,
        get_module: GetModuleOutput = get_layer_output,
    ):
        """
        Steer the hidden states of a layer using a steering vector.

        Args:
            layers: The layer(s) to steer
            steering_vector: The steering vector to apply
            factor: The factor to multiply the steering vector by
            positions: The position to steer. If None, all positions are steered.
            get_module: Function to get the module output to steer
        """
        if isinstance(layers, int):
            layers = [layers]
        for layer in layers:
            layer_device = get_layer_output(self, layer).device
            get_module(self, layer)[:, positions] += factor * steering_vector.to(
                layer_device
            )
