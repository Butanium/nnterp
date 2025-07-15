from __future__ import annotations
from typing import Callable
from loguru import logger
import torch as th
from torch.nn import Module
from torch import Size
from nnsight import LanguageModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import (
    TraceTensor,
    DummyCache,
    warn_about_status,
)
from .rename_utils import (
    get_rename_dict,
    LayerAccessor,
    IOType,
    get_ignores,
    mlp_returns_tuple,
    check_model_renaming,
    AttentionProbabilitiesAccessor,
    get_num_attention_heads,
    get_hidden_size,
    RenameConfig,
)

GetLayerObject = Callable[[int], TraceTensor]


def get_layer_output(model: StandardizedTransformer, layer: int) -> TraceTensor:
    return model.layers_output[layer]


class StandardizedTransformer(LanguageModel):
    """
    Renames the LanguageModel modules to match a standardized architecture.

    The model structure is organized as follows::

        StandardizedTransformer
        ├── layers
        │   ├── self_attn
        │   └── mlp
        ├── ln_final
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
        check_renaming (bool, default True): If True, the renaming of modules is validated.
            Defaults to True.
        allow_dispatch (bool, default True): If True, allows using trace() to dispatch the model
            when scan() fails during renaming checks. Defaults to True. You should set this to false
            if you plan to use the model remotely.
        check_attn_probs_with_trace (bool, default True): If True, the model will be dispatched and a test will ensure that the attention probabilities returned sum to 1.
        mlp_returns_tuple (bool, default False): Set to true if your model returns a tuple from the mlp. This is the case for most MoEs.
        attn_rename (str, optional): Extra module name to rename to ``self_attn``.
        mlp_rename (str, optional): Extra module name to rename to ``mlp``.
        ln_final_rename (str, optional): Extra module name to rename to ``ln_final``.
        lm_head_rename (str, optional): Extra module name to rename to ``lm_head``.
        model_rename (str, optional): Extra module name to rename to ``model``.
        layers_rename (str, optional): Extra module name to rename to ``layers``.
    """

    def __init__(
        self,
        model: str | Module,
        trust_remote_code: bool = False,
        check_renaming: bool = True,
        allow_dispatch: bool = True,
        check_attn_probs_with_trace: bool = True,
        rename_config: RenameConfig | None = None,
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
                logger.warning(
                    f"Attention implementation {impl} is not supported for attention pattern tracing. Please use eager attention implementation if you plan to access attention patterns."
                )
        else:
            logger.info(
                "Enforcing eager attention implementation for attention pattern tracing. The HF default would be to use sdpa if available. To use sdpa, set attn_implementation='sdpa' or None to use the HF default."
            )
            impl = "eager"
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        rename = get_rename_dict(rename_config=rename_config)
        user_rename = kwargs.pop("rename", None)
        if user_rename is not None:
            logger.info(
                f"Updating default rename with user-provided rename: {user_rename}"
            )
            rename.update(user_rename)
        super().__init__(
            model,
            attn_implementation=impl,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            rename=rename,
            **kwargs,
        )
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.__class__.__name__

        ignores = get_ignores(self._model, rename_config)

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
            returns_tuple=mlp_returns_tuple(self._model, rename_config),
        )

        self.num_layers = len(self.layers)
        self.num_heads = get_num_attention_heads(
            self._model, raise_error=False, rename_config=rename_config
        )
        self.hidden_size = get_hidden_size(
            self._model, raise_error=False, rename_config=rename_config
        )

        if check_renaming:
            check_model_renaming(self, model_name, ignores, allow_dispatch)
        self.attention_probabilities = AttentionProbabilitiesAccessor(
            self, rename_config=rename_config
        )
        if check_renaming:
            try:
                self.attention_probabilities.check_source(
                    allow_dispatch=allow_dispatch,
                    use_trace=check_attn_probs_with_trace,
                )
            except Exception as e:
                logger.error(
                    f"Attention probabilities is not available for {model_name} architecture. Disabling it. Error:\n{e}"
                )
                self.attention_probabilities.disable()
        warn_about_status(model_name, self._model, model_name)
        self._add_prefix_false_tokenizer = None

    @property
    def add_prefix_false_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._add_prefix_false_tokenizer is None:
            self._add_prefix_false_tokenizer = AutoTokenizer.from_pretrained(
                self.name_or_path, add_prefix_space=False
            )
        return self._add_prefix_false_tokenizer

    @property
    def attn_probs_available(self) -> bool:
        return self.attention_probabilities.enabled

    @property
    def input_size(self) -> Size:
        return self.inputs[1]["input_ids"].shape

    @property
    def logits(self) -> TraceTensor:
        """Returns the lm_head output"""
        return self.output.logits

    @property
    def next_token_probs(self) -> TraceTensor:
        return self.logits[:, -1, :].softmax(-1)

    def skip_layer(self, layer: int, skip_with: TraceTensor | None = None):
        """
        Skip the computation of a layer.

        Args:
            layer: The layer to skip
            skip_with: The input to skip the layer with. If None, the input of the layer is used.
        """
        return self.skip_layers(layer, layer, skip_with)

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
        if skip_with is None:
            skip_with = self.layers_input[start_layer]
        for layer in range(start_layer, end_layer):
            self.layers[layer].skip((skip_with, DummyCache()))
        self.layers[end_layer].skip((skip_with, DummyCache()))

    def steer(
        self,
        layers: int | list[int],
        steering_vector: th.Tensor,
        factor: float = 1,
        positions: int | list[int] | th.Tensor | None = None,
        get_layer_object_to_steer: GetLayerObject | None = None,
    ):
        """
        Steer the hidden states of a layer using a steering vector.

        Args:
            layers: The layer(s) to steer
            steering_vector: The steering vector to apply
            factor: The factor to multiply the steering vector by
            positions: The position to steer. If None, all positions are steered.
            get_layer_object_to_steer: Function that given a layer index, returns the object to steer in the model's. Default to model.layers_output[layer]
        """
        if get_layer_object_to_steer is None:
            get_layer_object_to_steer = self.layers_output
        if isinstance(layers, int):
            layers = [layers]
        for layer in layers:
            layer_device = get_layer_object_to_steer(layer).device
            get_layer_object_to_steer(layer)[
                :, positions
            ] += factor * steering_vector.to(layer_device)

    def project_on_vocab(self, h: TraceTensor) -> TraceTensor:
        h = self.ln_final(h)
        return self.lm_head(h)
