from __future__ import annotations

import torch as th
from nnsight_utils import LanguageModel, InterventionProxy, Envoy


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
) -> dict[str, str]:

    rename_dict = (
        {name: "self_attn" for name in ATTENTION_NAMES}
        | {name: "model" for name in MODEL_NAMES}
        | {name: "layers" for name in LAYER_NAMES}
        | {name: "ln_final" for name in LN_NAMES}
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
    update_rename_dict("ln_final", ln_final_name)
    update_rename_dict("lm_head", lm_head_name)

    return rename_dict


class StandardizedTransformer(LanguageModel):
    """

    StandardizedTransformer implementation for Hugging Face models.
    Renames the model modules to match the following architecture:
    StandardizedTransformer
        model
            layers
                self_attn
                mlp
            ln_final
        lm_head
    It already renames most models correctly, but you can add extra renaming rules.
    It also contains built-in functions to e.g. get the layer input, output, etc.


    Args:
        model_name: Name of the model to load.
        no_space_on_bos: If True
        attn_name: Extra module names to rename to self_attn.
        mlp_name: Extra module names to rename to mlp.
        ln_final_name: Extra module names to rename to ln_final.
        lm_head_name: Extra module names to rename to lm_head.
    """


    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        attn_name: str | None = None,
        mlp_name: str | None = None,
        ln_final_name: str | None = None,
        lm_head_name: str | None = None,
        **kwargs,
    ):
        kwargs.setdefault("torch_dtype", th.float16)

        kwargs.setdefault("device_map", "auto")

        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        super().__init__(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            rename_modules_dict=get_rename_dict(
                attn_name=attn_name,
                mlp_name=mlp_name,
                ln_final_name=ln_final_name,
                lm_head_name=lm_head_name,
            ),
            **kwargs,
        )
        try:
            model = self.model
        except

    def get_num_layers(self) -> int:
        return len(self.model.layers)

    def get_layer(self, layer: int) -> Envoy:
        return self.model.layers[layer]

    def get_layer_input(self, layer: int) -> InterventionProxy:
        return self.model.layers[layer].input[0][0]

    def get_layer_output(self, layer: int) -> InterventionProxy:
        return self.model.layers[layer].output[0]

    def get_attention(self, layer: int) -> Envoy:
        return self.model.layers[layer].self_attn

    def get_attention_output(self, layer: int) -> InterventionProxy:
        return self.model.layers[layer].self_attn.output[0]

    def get_logits(self) -> InterventionProxy:
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
