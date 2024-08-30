from __future__ import annotations
from dataclasses import dataclass
import torch as th
from torch.utils.data import DataLoader
from warnings import warn
from .nnsight_utils import (
    get_layer,
    get_layer_output,
    get_layer_input,
    get_attention,
    get_attention_output,
    get_next_token_probs,
    collect_activations,
    collect_activations_batched,
    get_num_layers,
    NNLanguageModel,
    GetModuleOutput,
    project_on_vocab,
)
from typing import Optional

__all__ = [
    "logit_lens",
    "TargetPrompt",
    "repeat_prompt",
    "TargetPromptBatch",
    "patchscope_lens",
    "patchscope_generate",
    "steer",
    "skip_layers",
    "patch_attention_lens",
    "patch_object_attn_lens",
    "object_lens",
]


@th.no_grad
def logit_lens(
    nn_model: NNLanguageModel, prompts: list[str] | str, scan=True, remote=False
):
    """
    Same as logit_lens but for Llama models directly instead of Transformer_lens models.
    Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

    Args:
        nn_model: NNSight Language Model
        prompts: List of prompts or a single prompt

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    with nn_model.trace(prompts, scan=scan, remote=remote) as tracer:
        hiddens_l = collect_activations(nn_model, prompts, open_context=False)
        probs_l = []
        for hiddens in hiddens_l:
            logits = project_on_vocab(nn_model, hiddens)
            probs = logits.softmax(-1).cpu()
            probs_l.append(probs)
        probs = th.stack(probs_l).transpose(0, 1).save()
    return probs.value


@dataclass
class TargetPrompt:
    prompt: str
    index_to_patch: int


def repeat_prompt(
    nn_model=None, words=None, rel=" ", sep="\n", placeholder="?"
) -> TargetPrompt:
    """
    Prompt used in the patchscopes paper to predict the next token.
    https://github.com/PAIR-code/interpretability/blob/master/patchscopes/code/next_token_prediction.ipynb
    """
    if words is None:
        words = [
            "king",
            "1135",
            "hello",
        ]
    assert nn_model is None or (
        len(nn_model.tokenizer.tokenize(placeholder)) == 1
    ), "Using a placeholder that is not a single token sounds like a bad idea"
    prompt = sep.join([w + rel + w for w in words]) + sep + placeholder
    index_to_patch = -1
    return TargetPrompt(prompt, index_to_patch)


@dataclass
class TargetPromptBatch:
    """
    A class to handle multiple target prompts with potentially different indices to patch
    """

    prompts: list[str]
    index_to_patch: th.Tensor

    @classmethod
    def from_target_prompts(cls, prompts_: list[TargetPrompt], tokenizer=None):
        prompts = [p.prompt for p in prompts_]
        index_to_patch = th.tensor([p.index_to_patch for p in prompts_])
        if index_to_patch.min() < 0:
            if tokenizer is None:
                raise ValueError(
                    "If using negative index_to_patch, a tokenizer must be provided"
                )
        return cls(prompts, index_to_patch)

    @classmethod
    def from_target_prompt(cls, prompt: TargetPrompt, batch_size: int):
        prompts = [prompt.prompt] * batch_size
        index_to_patch = th.tensor([prompt.index_to_patch] * batch_size)
        return cls(prompts, index_to_patch)

    @classmethod
    def from_prompts(
        cls, prompts: str | list[str], index_to_patch: int | list[int] | th.Tensor
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(index_to_patch, int):
            index_to_patch = th.tensor([index_to_patch] * len(prompts))
        elif isinstance(index_to_patch, list):
            index_to_patch = th.tensor(index_to_patch)
        elif not isinstance(index_to_patch, th.Tensor):
            raise ValueError(
                f"index_to_patch must be an int, a list of ints or a tensor, got {type(index_to_patch)}"
            )
        return cls(prompts, index_to_patch)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return TargetPrompt(self.prompts[idx], self.index_to_patch[idx])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def auto(
        target_prompt: str | TargetPrompt | list[TargetPrompt] | TargetPromptBatch,
        batch_size: int,
    ):
        if isinstance(target_prompt, TargetPrompt):
            target_prompt = TargetPromptBatch.from_target_prompt(
                target_prompt, batch_size
            )
        elif isinstance(target_prompt, list):
            target_prompt = TargetPromptBatch.from_target_prompts(target_prompt)
        elif not isinstance(target_prompt, TargetPromptBatch):
            raise ValueError(
                f"patch_prompts must be a str, a TargetPrompt, a list of TargetPrompt or a TargetPromptBatch, got {type(target_prompt)}"
            )
        return target_prompt


@th.no_grad
def patchscope_lens(
    nn_model: NNLanguageModel,
    source_prompts: list[str] | str | None = None,
    target_patch_prompts: (
        TargetPromptBatch | list[TargetPrompt] | TargetPrompt | None
    ) = None,
    layers=None,
    latents=None,
    remote=False,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight TL model
        source_prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompts: TargetPrompt(s) / TargetPromptBatch containing the prompt to patch and the index of the token to patch
        layers: List of layers to intervene on. If None, all layers are intervened on.
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if target_patch_prompts is None:
        target_patch_prompts = repeat_prompt()
    if latents is not None:
        if len(set([len(h) for h in latents])) > 1:
            raise ValueError("Inconsistent number of hiddens")
        num_sources = len(latents[0])
    else:
        if source_prompts is None:
            raise ValueError("Either source_prompts or hiddens must be provided")
        if isinstance(source_prompts, str):
            source_prompts = [source_prompts]
        num_sources = len(source_prompts)
    target_patch_prompts = TargetPromptBatch.auto(target_patch_prompts, num_sources)
    if len(target_patch_prompts) != num_sources:
        raise ValueError(
            f"Number of sources ({num_sources}) does not match number of patch prompts ({len(target_patch_prompts)})"
        )
    if latents is None:
        latents = collect_activations(nn_model, source_prompts, remote=remote)
    elif source_prompts is not None:
        raise ValueError("You cannot provide both source_prompts and hiddens")

    probs_l = []
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    for layer in layers:
        with nn_model.trace(
            target_patch_prompts.prompts,
            scan=layer == 0,
            remote=remote,
        ):
            get_layer_output(nn_model, layer)[
                th.arange(num_sources), target_patch_prompts.index_to_patch
            ] = latents[layer]
            probs_l.append(get_next_token_probs(nn_model).cpu().save())
    probs = th.cat([p.value for p in probs_l], dim=0)
    return probs.reshape(len(layers), num_sources, -1).transpose(0, 1)


@th.no_grad
def patchscope_generate(
    nn_model: NNLanguageModel,
    prompts: list[str] | str,
    target_patch_prompt: TargetPrompt,
    max_length: int = 50,
    layers=None,
    remote=False,
    max_batch_size=32,
):
    """
    Replace the hidden state of the patch_prompt.index_to_patch token in the patch_prompt.prompt with the hidden state of the last token of each prompt at each layer.
    Returns the probabilities of the next token in patch_prompt for each prompt for each layer intervention.
    Args:
        nn_model: The NNSight LanguageModel with llama architecture
        prompts: List of prompts or a single prompt to get the hidden states of the last token
        target_patch_prompt: A TargetPrompt object containing the prompt to patch and the index of the token to patch
        layers: List of layers to intervene on. If None, all layers are intervened on.
        max_length: The maximum length of the generated sequence
        remote: If True, the function will run on the nndif server. See `nnsight.net/status` to check which models are available.
        max_batch_size: The maximum number of prompts to intervene on at once.

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if len(prompts) > max_batch_size:
        warn(
            f"Number of prompts ({len(prompts)}) exceeds max_batch_size ({max_batch_size}). This may cause memory errors."
        )
    hiddens = collect_activations(nn_model, prompts, remote=remote, layers=layers)
    generations = {}
    gen_kwargs = dict(remote=remote, max_new_tokens=max_length)
    layer_loader = DataLoader(layers, batch_size=max(max_batch_size // len(prompts), 1))
    for layer_batch in layer_loader:
        with nn_model.generate(**gen_kwargs) as tracer:
            for layer in layer_batch:
                layer = layer.item()
                with tracer.invoke(
                    [target_patch_prompt.prompt] * len(prompts),
                    scan=layer == 0,
                ):
                    get_layer_output(nn_model, layer)[
                        :, target_patch_prompt.index_to_patch
                    ] = hiddens[layer]
                    gen = nn_model.generator.output.save()
                    generations[layer] = gen
    for k, v in generations.items():
        generations[k] = v.cpu()
    return generations


def steer(
    nn_model: NNLanguageModel,
    layers: int | list[int],
    steering_vector: th.Tensor,
    factor: float = 1,
    position: int = -1,
    get_module: GetModuleOutput = get_layer_output,
):
    """
    Steer the hidden states of a layer using a steering vector
    Args:
        nn_model: The NNSight model
        layers: The layer(s) to steer
        steering_vector: The steering vector to apply
        factor: The factor to multiply the steering vector by
    """
    if isinstance(layers, int):
        layers = [layers]
    for layer in layers:
        get_module(nn_model, layer)[:, position] += factor * steering_vector


def skip_layers(
    nn_model: NNLanguageModel,
    layers_to_skip: int | list[int],
    position: int = -1,
):
    """
    Skip the computation of the specified layers
    Args:
        nn_model: The NNSight model
        layers_to_skip: The layers to skip
    """
    if isinstance(layers_to_skip, int):
        layers_to_skip = [layers_to_skip]
    for layer in layers_to_skip:
        get_layer_output(nn_model, layer)[:, position] = get_layer_input(
            nn_model, layer
        )[:, position]


def patch_object_attn_lens(
    nn_model: NNLanguageModel,
    source_prompts: list[str] | str,
    target_prompts: list[str] | str,
    attn_idx_patch: int,
    num_patches: int = 5,
    scan=True,
):
    """
    A complex lens that makes the model attend to the hidden states of the last token of the source prompts instead of the attn_idx_patch token of the target prompts at last token prediction. For each layer, this intervention is performed for num_patches layers.
    Args:
        nn_model: The NNSight model
        source_prompts: The prompts to get the hidden states of the last token from
        target_prompts: The prompts to predict the next token for
        attn_idx_patch: The index of the token to patch in the target prompts
        num_patches: The number of layers to patch for each layer

    Returns:
        A tensor of shape (num_target_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each target prompt at each layer. Tensor is on the CPU.
    """
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    global probs_l
    num_layers = get_num_layers(nn_model)
    probs_l = []

    def get_act(model, layer):
        return get_attention(model, layer).input[1]["hidden_states"]

    source_hiddens = collect_activations(
        nn_model,
        source_prompts,
        get_activations=get_act,
    )
    for layer in range(num_layers):
        with nn_model.trace(target_prompts, scan=layer == 0 and scan):
            for next_layer in range(layer, min(num_layers, layer + num_patches)):
                get_attention(nn_model, next_layer).input[1]["hidden_states"][
                    :, attn_idx_patch
                ] = source_hiddens[next_layer]
            probs = get_next_token_probs(nn_model).cpu().save()
            probs_l.append(probs)
    return (
        th.cat([p.value for p in probs_l], dim=0)
        .reshape(num_layers, len(target_prompts), -1)
        .transpose(0, 1)
    )


@dataclass
class LatentPrompt:
    """
    A class to handle prompts with latent spots that will be replaced with latent vectors
    """

    prompt: str
    latent_spots: list[int]

    @classmethod
    def from_string(cls, prompt: str, tokenizer, placeholder_token: str | None = None):
        """
        Create a LatentPrompt object from a string prompt

        Args:
            prompt: The prompt string
            tokenizer: The tokenizer to use
            placeholder_token: The token to use as a placeholder. If None, the tokenizer's bos_token is used.
        """
        if placeholder_token is None:
            placeholder_token = tokenizer.bos_token
        tokens = tokenizer.tokenize(prompt)
        latent_spots = [
            i - len(tokens) for i, t in enumerate(tokens) if t == placeholder_token
        ]
        return cls(prompt, latent_spots)


@dataclass
class LatentPromptBatch:
    """
    A class to batch multiple LatentPrompt objects and modify them at the token level
    """

    inputs: dict
    latent_prompts: list[LatentPrompt]

    @classmethod
    def from_latent_prompts(cls, latent_prompts: list[LatentPrompt], tokenizer):
        prompts = [lp.prompt for lp in latent_prompts]
        inputs = tokenizer(prompts, return_tensors="pt")
        return cls(inputs, latent_prompts)

    def replace_tokens(self, token: int, replacements: list[int] | int):
        for tokens in self.inputs.input_ids:
            for i, t in enumerate(tokens):
                if t == token:
                    if isinstance(replacements, int):
                        tokens[i] = replacements
                    else:
                        tokens[i] = replacements.pop(0)
        return self

    def replace_spot_tokens(self, replacements: list[int] | int):
        for tokens, lp in zip(self.inputs.input_ids, self.latent_prompts):
            for spot in lp.latent_spots:
                if isinstance(replacements, int):
                    tokens[spot] = replacements
                else:
                    tokens[spot] = replacements.pop(0)
        return self


def run_latent_prompt(
    nn_model: NNLanguageModel,
    latent_prompts: list[LatentPrompt] | LatentPrompt | LatentPromptBatch,
    prompts: list[str] | str | None = None,
    latents: list[th.Tensor] | th.Tensor | None = None,
    collect_from_single_layer: int | bool = False,  # todo doc and ifs
    patch_from_layer: int = 0,
    patch_until_layer: int | None = None,
    remote=False,
    scan=True,
    batch_size=32,
):
    """
    Perform a forward pass on latent prompts and return the probabilities of the next token for each latent prompt.
    Args:
        nn_model: The NNSight model
        latent_prompts: A (list of) LatentPrompt object(s) / a latent prompt batch to run the forward pass on.
        prompts: The prompts to use as placeholders for the latent spots. If None, latents must be provided.
        latents: The latent vectors to use.  Must be of shape (num_latent_prompts, num_patches, hidden_size) if collect_from_single_layer is False, else (1, num_patches, hidden_size).
        If None, prompts must be provided.
        collect_from_single_layer: If True, assume that the latents are collected from a single layer. If int, will use latent from this layer for every patch.
        Must be of shape (1, num_patches, hidden_size).
        patch_from_layer: The layer to start patching from
        patch_until_layer: The layer to patch until. If None, all layers from patch_from_layer to the last layer are patched.
        remote: Whether to run the model on the remote device.
        scan: Whether to use nnsight's scan when tracing the model.

    Returns:
        The probabilities of the next token for each latent prompt of shape (num_latent_prompts, vocab_size)
    """
    if patch_until_layer is None:
        patch_until_layer = get_num_layers(nn_model) - 1
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(latent_prompts, LatentPrompt):
        latent_prompts = [latent_prompts]
    if isinstance(latent_prompts, LatentPromptBatch):
        inputs = latent_prompts.inputs
        latent_prompts = latent_prompts.latent_prompts
    else:
        inputs = [lp.prompt for lp in latent_prompts]
    if latents is None == prompts is None:
        raise ValueError("Either prompts or latents must be provided")
    if collect_from_single_layer is True and latents is None:
        raise ValueError(
            "When collecting from a single layer, latents must be provided"
        )
    if latents is not None and latents.dim() != 3:
        raise ValueError(
            f"Latents must be of shape (num_layers, num_patches, hidden_size), got {latents.shape}"
        )
    if collect_from_single_layer and latents is not None:
        if latents.shape[0] != 1:
            raise ValueError(
                f"Latents must be of shape (1, num_patches, hidden_size) when collect_from_single_layer is True, got {latents.shape}"
            )
    n_patches = len(prompts) if prompts is not None else latents.shape[1]
    num_spots = sum([len(lp.latent_spots) for lp in latent_prompts])
    if num_spots != n_patches:
        raise ValueError(
            f"Number of latent spots does not match number of prompts/latents: got {num_spots} spots and {n_patches} prompts/latents"
        )
    if latents is None:
        prompt_loader = DataLoader(prompts, batch_size=batch_size)
        latents = [[] for _ in range(patch_until_layer + 1)]
        for prompt_batch in prompt_loader:
            acts = collect_activations(
                nn_model,
                prompt_batch,
                layers=(
                    list(range(patch_from_layer, patch_until_layer + 1))
                    if not collect_from_single_layer
                    else collect_from_single_layer
                ),
                remote=remote,
            )  # [layer, batch, d]
            for layer, act in enumerate(acts):
                latents[layer].extend(act)

    with nn_model.trace(inputs, scan=scan, remote=remote):
        h_index = 0
        for i, lp in enumerate(latent_prompts):
            for spot in lp.latent_spots:
                for layer in range(patch_from_layer, patch_until_layer + 1):
                    get_layer_output(nn_model, layer)[i, spot] = (
                        latents[layer][h_index]
                        if not collect_from_single_layer
                        else latents[0][h_index]
                    )
                h_index += 1
        probs = get_next_token_probs(nn_model).cpu().save()
    return probs.value


def latent_prompt_lens(
    nn_model: NNLanguageModel,
    latent_prompts: list[LatentPrompt] | LatentPrompt,
    prompts: list[str] | str | None = None,
    latents: list[th.Tensor] | th.Tensor | None = None,
    collect_from_single_layer: bool = True,
    patch_from_layer: int | None = 0,
    patch_until_layer: int | None = None,
    layers=None,
    remote=False,
    scan=True,
    batch_size=32,
):
    if not collect_from_single_layer and patch_until_layer is not None:
        raise ValueError(
            "When collecting from multiple layers, patch_until_layer must be None"
        )
    if prompts is None and latents is None:
        raise ValueError("Either prompts or latents must be provided")
    if prompts is not None and latents is not None:
        raise ValueError("Only one of prompts or latents can be provided")
    if isinstance(prompts, str):
        prompts = [prompts]
    if prompts is not None:
        latents = collect_activations_batched(
            nn_model,
            prompts,
            remote=remote,
            batch_size=batch_size,
        )

    probs = []
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    for layer in layers:
        if collect_from_single_layer:
            latents_ = latents[layer].unsqueeze(0)
            if patch_until_layer is None:
                patch_until_layer_ = layer
            else:
                patch_until_layer_ = patch_until_layer
        else:
            patch_until_layer_ = layer
            latents_ = latents
        if patch_from_layer is None:
            patch_from_layer_ = layer
        else:
            patch_from_layer_ = patch_from_layer
        probs.append(
            run_latent_prompt(
                nn_model,
                latent_prompts,
                latents=latents_,
                collect_from_single_layer=collect_from_single_layer,
                patch_from_layer=patch_from_layer_,
                patch_until_layer=patch_until_layer_,
                remote=remote,
                scan=scan and layer == 0,
            )
        )
    return th.stack(probs).transpose(0, 1)


class Intervention:
    """
    A class that contains an intervention on a model
    """

    latent: th.Tensor
    layer: int
    position: int | None = None
    get_output: GetModuleOutput | None = None

    def __post_init__(self):
        if self.get_output is None:
            self.get_output = get_layer_output

    def apply(self, nn_model: NNLanguageModel):
        """
        Perform the intervention on the model
        """
        self.get_output(nn_model, self.layer)[:, self.position] = self.latent

    @classmethod
    def from_prompts(
        cls, nn_model, prompts, layers, position=None, remote=False, get_output=None
    ):
        if isinstance(layers, int):
            layers = [layers]
        hiddens = collect_activations(
            nn_model, prompts, layers, remote=remote, get_activations=get_layer_output
        )
        return [cls(h, l, position, get_output) for h, l in zip(hiddens, layers)]
