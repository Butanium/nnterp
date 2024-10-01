from __future__ import annotations

import torch as th
from transformers import AutoTokenizer
from nnsight.models.UnifiedTransformer import UnifiedTransformer
from nnsight.models.LanguageModel import LanguageModel, LanguageModelProxy
from nnsight.envoy import Envoy
from model_renaming import rename_model_modules


class StandardizedTransformer:
    """
    A standardized interface for transformer models, providing consistent access to model components
    regardless of the underlying implementation (TransformerLens or Hugging Face).
    """

    @classmethod
    def load(
        cls,
        model_name: str,
        use_tl: bool = False,
        no_space_on_bos: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> StandardizedTransformer:
        """
        Load a transformer model with the specified configuration.

        Args:
            model_name (str): The name or path of the model to load.
            use_tl (bool): If True, load a TransformerLens model; otherwise, load a Hugging Face model.
            no_space_on_bos (bool): If True, configure the tokenizer to not add a space before the beginning of sentence token.
            trust_remote_code (bool): If True, allow loading of remote code for the model and tokenizer.
            **kwargs: Additional keyword arguments to pass to the model loader.

        Returns:
            StandardizedTransformer: An instance of the appropriate StandardizedTransformer subclass.
        """
        if use_tl:
            return StandardizedTransformerTL.load(
                model_name, no_space_on_bos, trust_remote_code, **kwargs
            )
        else:
            return StandardizedTransformerHF.load(
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

    @classmethod
    def load(
        cls,
        model_name: str,
        no_space_on_bos: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> StandardizedTransformerTL:
        """
        Load a TransformerLens model with the specified configuration.

        Args:
            model_name (str): The name or path of the model to load.
            no_space_on_bos (bool): If True, configure the tokenizer to not add a space before the beginning of sentence token.
            trust_remote_code (bool): If True, allow loading of remote code for the model and tokenizer.
            **kwargs: Additional keyword arguments to pass to the model loader.

        Returns:
            StandardizedTransformerTL: An instance of StandardizedTransformerTL.
        """
        kwargs.setdefault("torch_dtype", th.float16)
        kwargs.setdefault(
            "n_devices", th.cuda.device_count() if th.cuda.is_available() else 1
        )
        kwargs.setdefault("device", "cuda" if th.cuda.is_available() else "cpu")
        kwargs["processing"] = False

        if no_space_on_bos:
            tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
            tokenizer_kwargs.update(
                dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            kwargs["tokenizer"] = tokenizer

        return cls(model_name, **kwargs)

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

    @classmethod
    def load(
        cls,
        model_name: str,
        no_space_on_bos: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> StandardizedTransformerHF:
        """
        Load a Hugging Face model with the specified configuration.

        Args:
            model_name (str): The name or path of the model to load.
            no_space_on_bos (bool): If True, configure the tokenizer to not add a space before the beginning of sentence token.
            trust_remote_code (bool): If True, allow loading of remote code for the model and tokenizer.
            **kwargs: Additional keyword arguments to pass to the model loader.

        Returns:
            StandardizedTransformerHF: An instance of StandardizedTransformerHF.
        """
        kwargs.setdefault("torch_dtype", th.float16)
        kwargs.setdefault("device_map", "auto")

        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
        if no_space_on_bos:
            tokenizer_kwargs.update(
                dict(add_prefix_space=False, trust_remote_code=trust_remote_code)
            )

        self = cls(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        rename_model_modules(self)
        return self

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
