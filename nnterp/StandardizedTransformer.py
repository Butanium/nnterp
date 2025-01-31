from __future__ import annotations

import torch as th
from nnsight_utils import LanguageModel, InterventionProxy, Envoy


# Names to rename in the model
attention_names = ["attn", "self_attention", "attention"]
model_names = ["transformer", "gpt_neox"]
layer_names = ["h"]
ln_names = ["final_layer_norm", "ln_f"]
lm_head_names = ["embed_out"]


llm_rename_dict = {}
for name in attention_names:
    llm_rename_dict[name] = "self_attn"
for name in model_names:
    llm_rename_dict[name] = "model"
for name in layer_names:
    llm_rename_dict[name] = "layers"
for name in ln_names:
    llm_rename_dict[name] = "ln_final"


class StandardizedTransformer(LanguageModel):
    """StandardizedTransformer implementation for Hugging Face models."""

    def __init__(
        self,
        model_name: str,
        no_space_on_bos: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("torch_dtype", th.float16)
        kwargs.setdefault("device_map", "auto")

        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        if no_space_on_bos:
            tokenizer_kwargs.update(
                dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
            )
        super().__init__(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            rename_modules_dict=llm_rename_dict,
            **kwargs,
        )

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
