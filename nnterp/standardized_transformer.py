from __future__ import annotations
from typing import Union
from enum import Enum
import torch as th
from nnsight import LanguageModel
from nnsight.intervention.tracing.globals import Object
from nnsight.intervention.envoy import Envoy

from .nnsight_utils import (
    TraceTensor,
    get_rename_dict,
    get_num_layers,
    get_layer,
    get_attention,
    get_mlp,
    get_logits,
    get_unembed_norm,
    project_on_vocab,
    skip_layer,
    skip_layers,
    get_next_token_probs,
    GetModuleOutput,
    get_layer_output,
)


class IOType(Enum):
    """Enum to specify input or output access"""

    INPUT = "input"
    OUTPUT = "output"


class ModuleAccessor:
    """Simple accessor for module access (no get/set, just module retrieval)"""

    def __init__(self, model, attr_name: str | None):
        self.model = model
        self.attr_name = attr_name

    def __getitem__(self, layer: int) -> Envoy:
        target = self.model.model.layers[layer]
        if self.attr_name is not None:
            target = getattr(target, self.attr_name)
        return target


class LayerAccessor(ModuleAccessor):
    """I/O accessor that inherits from ModuleAccessor and provides input/output access with setter"""

    def __init__(
        self, model, attr_name: str | None, io_type: IOType, returns_tuple: bool = False
    ):
        super().__init__(model, attr_name)
        self.io_type = io_type
        self.returns_tuple = returns_tuple

    def __getitem__(self, layer: int) -> TraceTensor:
        # Get the module using parent class
        module = super().__getitem__(layer)

        # Get input or output
        if self.io_type == IOType.INPUT:
            target = module.input
        else:  # IOType.OUTPUT
            target = module.output
        if self.returns_tuple:
            return target[0]
        else:
            return target

    def __setitem__(self, layer: int, value: TraceTensor):
        # Get the module using parent class
        module = super().__getitem__(layer)

        # Set input or output
        if self.io_type == IOType.INPUT:
            if self.returns_tuple:
                module.input = (value,)
            else:
                module.input = value
        else:
            if self.returns_tuple:
                module.output = (value,)
            else:
                module.output = value


class StandardizedTransformer(LanguageModel):
    """
    Renames the LanguageModel modules to match a standardized architecture.

    The model structure is organized as follows::

        StandardizedTransformer
            ├── model
            │   ├── layers
            │   │   ├── self_attn
            │   │   └── mlp
            │   └── ln_final
            └── lm_head

    In addition to renaming modules, this class provides built-in accessors to extract and set intermediate activations:

    - layers_output[i]: Get/set layer output at layer i
    - layers_input[i]: Get/set layer input at layer i
    - layers[i]: Get layer module at layer i
    - attention_output[i]: Get/set attention output at layer i
    - attentions[i]: Get attention module at layer i
    - mlp_output[i]: Get/set MLP output at layer i
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
        **kwargs,
    ):
        kwargs.setdefault("torch_dtype", th.float16)
        kwargs.setdefault("device_map", "auto")

        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        super().__init__(
            repo_id,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            rename=get_rename_dict(
                attn_name=attn_rename,
                mlp_name=mlp_rename,
                ln_final_name=ln_final_rename,
                lm_head_name=lm_head_rename,
                model_name=model_rename,
                layers_name=layers_rename,
            ),
            **kwargs,
        )

        if check_renaming:
            self._check_renaming(repo_id)
        self.num_layers = get_num_layers(self)

        # Create accessor instances
        self.layers_output = LayerAccessor(
            self, None, IOType.OUTPUT, returns_tuple=True
        )
        self.layers_input = LayerAccessor(self, None, IOType.INPUT, returns_tuple=False)
        self.layers = ModuleAccessor(self, None)
        self.attention_output = LayerAccessor(
            self, "self_attn", IOType.OUTPUT, returns_tuple=True
        )
        self.attentions = ModuleAccessor(self, "self_attn")
        self.mlp_output = LayerAccessor(self, "mlp", IOType.OUTPUT, returns_tuple=False)
        self.mlps = ModuleAccessor(self, "mlp")

    @property
    def unembed_norm(self) -> Envoy:
        return get_unembed_norm(self)

    def project_on_vocab(self, h: TraceTensor) -> TraceTensor:
        return project_on_vocab(self, h)

    def get_logits(self) -> TraceTensor:
        """Returns the lm_head output"""
        return get_logits(self)

    def get_next_token_probs(self) -> TraceTensor:
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

    def _check_renaming(self, repo_id: str):
        try:
            _ = self.model
        except AttributeError as exc:
            raise ValueError(
                f"Could not find model module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the model module to the model_rename argument."
            ) from exc
        try:
            _ = self.model.layers
        except AttributeError as exc:
            raise ValueError(
                f"Could not find layers module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the layers module to the layers_rename argument."
            ) from exc
        try:
            _ = get_unembed_norm(self)
        except AttributeError as exc:
            raise ValueError(
                f"Could not find norm module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the norm module to the ln_final_rename argument."
            ) from exc
        try:
            _ = self.lm_head
        except AttributeError as exc:
            raise ValueError(
                f"Could not find lm_head module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the lm_head module to the lm_head_rename argument."
            ) from exc
        try:
            _ = get_attention(self, 0)
        except AttributeError as exc:
            raise ValueError(
                f"Could not find self_attn module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the self_attn module to the attn_rename argument."
            ) from exc
        try:
            _ = get_mlp(self, 0)
        except AttributeError as exc:
            raise ValueError(
                f"Could not find mlp module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the mlp module to the mlp_rename argument."
            ) from exc
