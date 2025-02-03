from __future__ import annotations
import torch as th
from .nnsight_utils import LanguageModel, InterventionProxy, Envoy


ATTENTION_NAMES = ["attn", "self_attention", "attention"]
MODEL_NAMES = ["transformer", "gpt_neox"]
LAYER_NAMES = ["h"]
LN_NAMES = ["final_layer_norm", "ln_f"]
LM_HEAD_NAMES = ["embed_out"]


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


class StandardizedTransformer(LanguageModel):
    """
    Renames the LanguageModel modules to match a standardized architecture.

    The model structure is organized as follows:
        StandardizedTransformer
            ├── model
            │   ├── layers
            │   │   ├── self_attn
            │   │   └── mlp
            │   └── ln_final
            └── lm_head

    In addition to renaming modules, this class provides built-in methods to extract intermediate activations, such as layer inputs, outputs, and attention outputs.

    Args:
        repo_id (str): Hugging Face repository ID or path of the model to load.
        trust_remote_code (bool, optional): If True, remote code will be trusted when
            loading the model. Defaults to False.
        attn_rename (str, optional): Extra module name to rename to `self_attn`.
        mlp_rename (str, optional): Extra module name to rename to `mlp`.
        ln_final_rename (str, optional): Extra module name to rename to `ln_final`.
        lm_head_rename (str, optional): Extra module name to rename to `lm_head`.
        model_rename (str, optional): Extra module name to rename to `model`.
        layers_rename (str, optional): Extra module name to rename to `layers`.
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
        self.num_layers = len(self.model.layers)

    def get_num_layers(self) -> int:
        return self.num_layers

    def get_layer(self, layer: int) -> Envoy:
        return self.model.layers[layer]

    def get_layer_input(self, layer: int) -> InterventionProxy:
        return self.model.layers[layer].input

    def get_layer_output(self, layer: int) -> InterventionProxy:
        return self.model.layers[layer].output[0]

    def get_attention(self, layer: int) -> Envoy:
        return self.model.layers[layer].self_attn

    def get_attention_output(self, layer: int) -> InterventionProxy:
        return self.model.layers[layer].self_attn.output[0]

    def get_mlp_output(self, layer: int) -> InterventionProxy:
        """
        Get the output of the MLP of a layer
        """
        return self.model.layers[layer].mlp.output

    def get_logits(self) -> InterventionProxy:
        """Returns the lm_head output"""
        return self.lm_head.output

    def get_unembed_norm(self) -> Envoy:
        return self.model.norm

    def get_unembed(self) -> Envoy:
        return self.lm_head

    def project_on_vocab(self, h: InterventionProxy) -> InterventionProxy:
        ln_out = self.model.norm(h)
        return self.lm_head(ln_out)

    def get_next_token_probs(self) -> InterventionProxy:
        return self.get_logits()[:, -1, :].softmax(-1)

    def stop_at_layer(self, layer: int) -> InterventionProxy:
        self.get_layer(layer).output.stop()

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
            _ = self.model.norm
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
            _ = self.model.layers[0].self_attn
        except AttributeError as exc:
            raise ValueError(
                f"Could not find self_attn module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the self_attn module to the attn_rename argument."
            ) from exc
        try:
            _ = self.model.layers[0].mlp
        except AttributeError as exc:
            raise ValueError(
                f"Could not find mlp module in {repo_id} architecture. This means that it was not properly renamed.\n"
                "Please pass the name of the mlp module to the mlp_rename argument."
            ) from exc
