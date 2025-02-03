from __future__ import annotations


from nnsight import LanguageModel
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.graph.proxy import InterventionProxy

import torch as th
from torch.utils.data import DataLoader
import nnsight as nns
from typing import Union, Callable
from contextlib import nullcontext
from transformers import AutoTokenizer
from .standardized_transformer import get_rename_dict

GetModuleOutput = Callable[[LanguageModel, int], InterventionProxy]


def load_model(
    model_name: str,
    trust_remote_code=False,
    use_tl=False,
    use_module_renaming=True,
    no_space_on_bos=False,
    rename_kwargs=None,
    **kwargs_,
):
    """
    Load a model into nnsight. If use_tl is True, a TransformerLens model is loaded.
    Default device is "auto" and default torch_dtype is th.float16.

    Args:
        no_space_on_bos: If True, add_prefix_space is set to False in the tokenizer. It is useful if you want to use the tokenizer to get the first token of a word when it's not after a space.
    """
    if rename_kwargs is None:
        rename_kwargs = {}
    kwargs = dict(torch_dtype=th.float16, trust_remote_code=trust_remote_code)
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
        rename_modules_dict = get_rename_dict(**rename_kwargs) if use_module_renaming else None
        model = LanguageModel(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
            rename=rename_modules_dict,
            **kwargs,
        )
        return model


def get_num_layers(nn_model: LanguageModel):
    """
    Get the number of layers in the model
    Args:
        nn_model: The NNSight model
    Returns:
        The number of layers in the model
    """
    return len(nn_model.model.layers)


def get_layer(nn_model: LanguageModel, layer: int) -> Envoy:
    """
    Get the layer of the model
    Args:
        nn_model: The NNSight model
        layer: The layer to get
    Returns:
        The Envoy for the layer
    """
    return nn_model.model.layers[layer]


def get_layer_input(nn_model: LanguageModel, layer: int) -> InterventionProxy:
    """
    Get the hidden state input of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the input of
    Returns:
        The Proxy for the input of the layer
    """
    return get_layer(nn_model, layer).input


def get_layer_output(nn_model: LanguageModel, layer: int) -> InterventionProxy:
    """
    Get the output of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the layer
    """
    output = get_layer(nn_model, layer).output
    return output[0]


def get_attention(nn_model: LanguageModel, layer: int) -> Envoy:
    """
    Get the attention module of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the attention module of
    Returns:
        The Envoy for the attention module of the layer
    """
    return nn_model.model.layers[layer].self_attn


def get_attention_output(nn_model: LanguageModel, layer: int) -> InterventionProxy:
    """
    Get the output of the attention block of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the attention block of the layer
    """
    output = get_attention(nn_model, layer).output
    return output[0]


def get_mlp_output(nn_model: LanguageModel, layer: int) -> InterventionProxy:
    """
    Get the output of the MLP of a layer
    """
    return get_layer(nn_model, layer).mlp.output

def get_logits(nn_model: LanguageModel) -> InterventionProxy:
    """
    Get the logits of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the logits of the model
    """
    return nn_model.lm_head.output


def get_unembed_norm(nn_model: LanguageModel) -> Envoy:
    """
    Get the last layer norm of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Envoy for the last layer norm of the model
    """
    return nn_model.model.norm


def get_unembed(nn_model: LanguageModel) -> Envoy:
    """
    Get the unembed module of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Envoy for the unembed module of the model
    """
    return nn_model.lm_head


def project_on_vocab(
    nn_model: LanguageModel, h: InterventionProxy
) -> InterventionProxy:
    """
    Project the hidden states on the vocabulary, after applying the model's last layer norm
    Args:
        nn_model: The NNSight model
        h: The hidden states to project
    Returns:
        The Proxy for the hidden states projected on the vocabulary
    """
    ln_out = nn_model.model.norm(h)
    return nn_model.lm_head(ln_out)


def get_next_token_probs(nn_model: LanguageModel) -> InterventionProxy:
    """
    Get the probabilities of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the probabilities of the model
    """
    return get_logits(nn_model)[:, -1, :].softmax(-1)


@th.no_grad
def collect_activations(
    nn_model: LanguageModel,
    prompts,
    layers=None,
    get_activations: GetModuleOutput | None = None,
    remote=False,
    idx: int | None = None,
    open_context=True,
):
    """
    Collect the hidden states of the last token of each prompt at each layer

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for
        open_context: Whether to open a trace context to collect activations. Set to false if you want to
            use this function in a context that already has a trace context open

    Returns:
        The hidden states of the last token of each prompt at each layer, moved to cpu. If open_context is False, returns a list of
        Proxies. Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if get_activations is None:
        get_activations = get_layer_output
    if idx is None:
        idx = -1
    if idx < 0 and nn_model.tokenizer.padding_side != "left":
        raise ValueError("negative index is currently only supported with left padding")
    if idx > 0 and nn_model.tokenizer.padding_side != "right":
        raise ValueError(
            "positive index is currently only supported with right padding"
        )
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    last_layer = max(layers)
    if min(layers) < 0:
        last_layer = max(last_layer, get_num_layers(nn_model) + min(layers))

    def wrap(h):
        if open_context:
            return h.cpu().save()
        return h

    # Collect the hidden states of the last token of each prompt at each layer
    context = nn_model.trace(prompts, remote=remote) if open_context else nullcontext()
    with context:
        acts = [wrap(get_activations(nn_model, layer)[:, idx]) for layer in layers]
        get_layer(nn_model, last_layer).output.stop()
        # This early stopping is useful to avoid e.g. Gemma2 converting its logits to floats
    return th.stack(acts)


@th.no_grad
def collect_activations_session(
    nn_model,
    prompts,
    batch_size,
    layers=None,
    get_activations=None,
    remote=False,
    idx=None,
):
    """
    Collect the hidden states of the specified token of each prompt at each layer in batches using a nnsight session.

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        batch_size: The batch size to use
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for. Default is -1 (last token).

    Returns:
        The hidden states of the specified token of each prompt at each layer, moved to cpu.
        Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    last_layer = max(layers)
    if min(layers) < 0:
        last_layer = max(last_layer, get_num_layers(nn_model) + min(layers))
    if get_activations is None:
        get_activations = get_layer_output
    if idx is None:
        idx = -1
    if idx < 0 and nn_model.tokenizer.padding_side != "left":
        raise ValueError("negative index is currently only supported with left padding")
    if idx > 0 and nn_model.tokenizer.padding_side != "right":
        raise ValueError(
            "positive index is currently only supported with right padding"
        )
    with nn_model.session(remote=remote) as session:
        all_acts = nns.list().save()
        dl = DataLoader(prompts, batch_size=batch_size)
        with session.iter(dl) as batch:
            with nn_model.trace(batch):
                acts = [
                    get_activations(nn_model, layer)[
                        :,
                        idx,
                    ]
                    .cpu()
                    .save()
                    for layer in layers
                ]
                get_layer(nn_model, last_layer).output.stop()
            all_acts.append(th.stack(acts).save())
        all_acts = nns.apply(th.cat, all_acts, dim=1).save()
    return all_acts.value


def collect_activations_batched(
    nn_model: LanguageModel,
    prompts,
    batch_size,
    layers=None,
    get_activations: GetModuleOutput | None = None,
    remote=False,
    idx=None,
    tqdm=None,
    use_session=True,
):
    """
    Collect the hidden states of the last token of each prompt at each layer in batches

    Args:
        nn_model: The NNSight model
        prompts: The prompts to collect activations for
        batch_size: The batch size to use
        layers: The layers to collect activations for, default to all layers
        get_activations: The function to get the activations, default to layer output
        remote: Whether to run the model on the remote device
        idx: The index of the token to collect activations for. Default is -1 (last token).
        tqdm: Whether to use tqdm to show progress, default to None (no progress bar)
        use_session: Whether to use a nnsight session to collect activations. Not sure why you'd want turn that off but who knows

    Returns:
        The hidden states of the specified token of each prompt at each layer, moved to cpu.
        Dimensions are (num_layers, num_prompts, hidden_size)
    """
    if use_session and remote:
        return collect_activations_session(
            nn_model,
            prompts,
            batch_size,
            layers,
            get_activations,
            remote,
            idx,
        )
    num_prompts = len(prompts)
    acts = []
    it = range(0, num_prompts, batch_size)
    if tqdm is not None:
        it = tqdm(it)
    for i in it:
        batch = prompts[i : min(i + batch_size, num_prompts)]
        acts_batch = collect_activations(
            nn_model, batch, layers, get_activations, remote, idx
        )
        acts.append(acts_batch)
    return th.cat(acts, dim=1)


def next_token_probs(
    nn_model: LanguageModel, prompt: str | list[str], remote=False
) -> th.Tensor:
    """
    Get the probabilities of the next token for the prompt
    Args:
        nn_model: The NNSight model
        prompt: The prompt to get the probabilities for
        remote: Whether to run the model on the remote device
    Returns:
        The probabilities of the next token for the prompt
    """
    with nn_model.trace(prompt, remote=remote):
        out = nn_model.output.logits
        out = out[:, -1].softmax(-1).cpu().save()
    return out.value

def stop_at_layer(nn_model: LanguageModel, layer: int) -> InterventionProxy:
    """
    Stop the output of the model at a given layer
    """
    return get_layer(nn_model, layer).output.stop()
