from __future__ import annotations

import torch as th
from transformers import AutoTokenizer
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from nnsight.models.LanguageModel import LanguageModel, LanguageModelProxy
from nnsight.envoy import Envoy


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
    

class StandardizedTransformer:
    """
    A standardized interface for transformer models, providing consistent access to model components
    regardless of the underlying implementation (TransformerLens or Hugging Face).
    """

    def __init__(
        self,
        model_name: str,
        use_tl: bool = False,
        no_space_on_bos: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize a transformer model with the specified configuration.
        """
        if use_tl:
            self.model = StandardizedTransformerTL(
                model_name, no_space_on_bos, trust_remote_code, **kwargs
            )
        else:
            self.model = StandardizedTransformerHF(
                model_name, no_space_on_bos, trust_remote_code, **kwargs
            )

    def get_num_layers(self) -> int:
        """Get the number of layers in the transformer model."""
        raise NotImplementedError

    def get_layer(self, layer: int) -> Envoy:
        """Get the Envoy object for a specific layer."""
        raise NotImplementedError

    def get_layer_input(self, layer: int) -> LanguageModelProxy:
        """Get the input to a specific layer."""
        raise NotImplementedError

    def get_layer_output(self, layer: int) -> LanguageModelProxy:
        """Get the output from a specific layer."""
        raise NotImplementedError

    def get_attention(self, layer: int) -> Envoy:
        """Get the attention module for a specific layer."""
        raise NotImplementedError

    def get_attention_output(self, layer: int) -> LanguageModelProxy:
        """Get the output from the attention module of a specific layer."""
        raise NotImplementedError

    def get_logits(self) -> LanguageModelProxy:
        """Get the logits (pre-softmax output) of the model."""
        raise NotImplementedError

    def get_unembed_norm(self) -> Envoy:
        """Get the normalization layer before the unembedding."""
        raise NotImplementedError

    def get_unembed(self) -> Envoy:
        """Get the unembedding layer of the model."""
        raise NotImplementedError

    def project_on_vocab(self, h: LanguageModelProxy) -> LanguageModelProxy:
        """Project hidden states onto the vocabulary space."""
        raise NotImplementedError

    def get_next_token_probs(self) -> LanguageModelProxy:
        """Get the probabilities for the next token."""
        raise NotImplementedError


class StandardizedTransformerTL(UnifiedTransformer, StandardizedTransformer):
    """StandardizedTransformer implementation for TransformerLens models."""

    def __init__(
        self,
        model_name: str,
        no_space_on_bos: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("device", "cuda" if th.cuda.is_available() else "cpu")
        kwargs["processing"] = False

        if no_space_on_bos:
            tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
            tokenizer_kwargs.update(
                dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            kwargs["tokenizer"] = tokenizer
        super().__init__(model_name, **kwargs)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    def get_layer(self, layer: int) -> Envoy:
        return self.blocks[layer]

    def get_layer_input(self, layer: int) -> LanguageModelProxy:
        return self.blocks[layer].input[0][0]

    def get_layer_output(self, layer: int) -> LanguageModelProxy:
        return self.blocks[layer].output

    def get_attention(self, layer: int) -> Envoy:
        return self.blocks[layer].attn

    def get_attention_output(self, layer: int) -> LanguageModelProxy:
        return self.blocks[layer].attn.output

    def get_logits(self) -> LanguageModelProxy:
        return self.unembed.output

    def get_unembed_norm(self) -> Envoy:
        return self.ln_final

    def get_unembed(self) -> Envoy:
        return self.unembed

    def project_on_vocab(self, h: LanguageModelProxy) -> LanguageModelProxy:
        ln_out = self.ln_final(h)
        return self.unembed(ln_out)

    def get_next_token_probs(self) -> LanguageModelProxy:
        return self.get_logits()[:, -1, :].softmax(-1)


class StandardizedTransformerHF(LanguageModel, StandardizedTransformer):
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

    def get_layer_input(self, layer: int) -> LanguageModelProxy:
        return self.model.layers[layer].input[0][0]

    def get_layer_output(self, layer: int) -> LanguageModelProxy:
        return self.model.layers[layer].output[0]

    def get_attention(self, layer: int) -> Envoy:
        return self.model.layers[layer].self_attn

    def get_attention_output(self, layer: int) -> LanguageModelProxy:
        return self.model.layers[layer].self_attn.output[0]

    def get_logits(self) -> LanguageModelProxy:
        return self.lm_head.output

    def get_unembed_norm(self) -> Envoy:
        return self.model.norm

    def get_unembed(self) -> Envoy:
        return self.lm_head

    def project_on_vocab(self, h: LanguageModelProxy) -> LanguageModelProxy:
        ln_out = self.model.norm(h)
        return self.lm_head(ln_out)

    def get_next_token_probs(self) -> LanguageModelProxy:
        return self.get_logits()[:, -1, :].softmax(-1)
