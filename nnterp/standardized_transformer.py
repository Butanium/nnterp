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
    RouterAccessor,
    get_num_attention_heads,
    get_hidden_size,
    RenameConfig,
)

GetLayerObject = Callable[[int], TraceTensor]


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
    - routers[i]: Get router module at layer i (MoE models only)
    - routers.router_outputs[i]: Get router output at layer i (MoE models only)
    - routers.router_weights[i]: Get router weights at layer i (MoE models only)

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
        rename_config (RenameConfig, default None): A RenameConfig object to use for renaming the model. If None, a default RenameConfig will be used.
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
        if "attn_implementation" in kwargs:
            impl = kwargs.pop("attn_implementation", None)
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
        
        # Initialize router accessor
        self.routers = RouterAccessor(self, rename_config=rename_config)
        if check_renaming and self.routers.enabled:
            try:
                self.routers.check_router_structure()
            except Exception as e:
                logger.error(
                    f"Router access is not available for {model_name} architecture. Disabling it. Error:\n{e}"
                )
                self.routers.disable()
        
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
    def routers_available(self) -> bool:
        return self.routers.enabled

    @property
    def input_ids(self) -> TraceTensor:
        return self.inputs[1]["input_ids"]

    @property
    def input_size(self) -> Size:
        """
        Returns the shape of the input tensor (batch_size, sequence_length)
        """
        return self.input_ids.shape

    @property
    def attention_mask(self) -> TraceTensor:
        """Returns the attention mask tensor."""
        return self.inputs[1]["attention_mask"]

    @property
    def logits(self) -> TraceTensor:
        """Returns the predicted logits."""
        return self.output.logits

    @property
    def next_token_probs(self) -> TraceTensor:
        """Returns the predicted probabilities for the next token.
        Assumes padding_side is "left"."""
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
        for layer in range(start_layer, end_layer + 1):
            self.layers[layer].skip((skip_with, DummyCache()))

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

    def project_on_vocab(self, hidden_state: TraceTensor) -> TraceTensor:
        hidden_state = self.ln_final(hidden_state)
        return self.lm_head(hidden_state)

    def probs_to_dict(self, tokens: th.Tensor, probs: th.Tensor) -> dict[str, float]:
        """
        Convert a tensor of probabilities to a dictionary mapping tokens to their probabilities
        """
        return {
            token: prob.item()
            for token, prob in zip(self.tokenizer.convert_ids_to_tokens(tokens), probs)
        }

    def get_topk_closest_tokens(
        self, hidden_state: th.Tensor, k=5
    ) -> dict[str, float] | list[dict[str, float]]:
        """
        Get the top-k closest tokens to the hidden state h.
        Args:
            h: The hidden state to project on the vocabulary. Shape (batch_size, hidden_size) or (hidden_size,).
            k: The number of top tokens to return.
            returns_df: If True, returns a DataFrame instead of a dictionary. Note that you need to have pandas installed for this to work.
                Pandas is included in ``pip install nnterp[display]``.
        Returns:
            A dictionary mapping tokens to their probabilities if h is 1D, or a list of dictionaries if h is 2D.
        """
        if hidden_state.shape[-1] != self.hidden_size and self.hidden_size is not None:
            raise ValueError(
                f"Hidden state shape {hidden_state.shape} does not match model hidden size {self.hidden_size}."
            )

        logits = self.project_on_vocab(hidden_state)
        probs = logits.softmax(-1)
        topk_tokens = th.topk(probs, k=k, dim=-1)
        if hidden_state.ndim == 1:
            return self.probs_to_dict(topk_tokens.indices, topk_tokens.values)
        elif hidden_state.ndim == 2:
            return [
                self.probs_to_dict(topk_tokens.indices[i], topk_tokens.values[i])
                for i in range(hidden_state.shape[0])
            ]
        else:
            raise ValueError(
                f"Unsupported hidden state shape {hidden_state.shape}. Expected 1D or 2D tensor."
            )
