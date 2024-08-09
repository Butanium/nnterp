from nnsight.models.UnifiedTransformer import UnifiedTransformer
from nnsight.models.LanguageModel import LanguageModelProxy, LanguageModel
from nnsight.envoy import Envoy
import torch as th
from torch.utils.data import DataLoader
from typing import Union, Callable
from contextlib import nullcontext

NNLanguageModel = Union[UnifiedTransformer, LanguageModel]
GetModuleOutput = Callable[[NNLanguageModel, int], LanguageModelProxy]


def get_num_layers(nn_model: NNLanguageModel):
    """
    Get the number of layers in the model
    Args:
        nn_model: The NNSight model
    Returns:
        The number of layers in the model
    """
    if isinstance(nn_model, UnifiedTransformer):
        return len(nn_model.blocks)
    else:
        return len(nn_model.model.layers)


def get_layer(nn_model: NNLanguageModel, layer: int) -> Envoy:
    """
    Get the layer of the model
    Args:
        nn_model: The NNSight model
        layer: The layer to get
    Returns:
        The Envoy for the layer
    """
    if isinstance(nn_model, UnifiedTransformer):
        return nn_model.blocks[layer]
    else:
        return nn_model.model.layers[layer]


def get_layer_input(nn_model: NNLanguageModel, layer: int) -> LanguageModelProxy:
    """
    Get the hidden state input of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the input of
    Returns:
        The Proxy for the input of the layer
    """
    return get_layer(nn_model, layer).input[0][0]


def get_layer_output(nn_model: NNLanguageModel, layer: int) -> LanguageModelProxy:
    """
    Get the output of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the layer
    """
    output = get_layer(nn_model, layer).output
    if isinstance(nn_model, UnifiedTransformer):
        return output
    else:
        return output[0]


def get_attention(nn_model: NNLanguageModel, layer: int) -> Envoy:
    """
    Get the attention module of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the attention module of
    Returns:
        The Envoy for the attention module of the layer
    """
    if isinstance(nn_model, UnifiedTransformer):
        return nn_model.blocks[layer].attn
    else:
        return nn_model.model.layers[layer].self_attn


def get_attention_output(nn_model: NNLanguageModel, layer: int) -> LanguageModelProxy:
    """
    Get the output of the attention block of a layer
    Args:
        nn_model: The NNSight model
        layer: The layer to get the output of
    Returns:
        The Proxy for the output of the attention block of the layer
    """
    output = get_attention(nn_model, layer).output
    if isinstance(nn_model, UnifiedTransformer):
        return output
    else:
        return output[0]


def get_logits(nn_model: NNLanguageModel) -> LanguageModelProxy:
    """
    Get the logits of the model
    Args:
        nn_model: The NNSight model
    Returns:
        The Proxy for the logits of the model
    """
    if isinstance(nn_model, UnifiedTransformer):
        return nn_model.unembed.output
    else:
        return nn_model.lm_head.output


def project_on_vocab(
    nn_model: NNLanguageModel, h: LanguageModelProxy
) -> LanguageModelProxy:
    """
    Project the hidden states on the vocabulary, after applying the model's last layer norm
    Args:
        nn_model: The NNSight model
        h: The hidden states to project
    Returns:
        The Proxy for the hidden states projected on the vocabulary
    """
    if isinstance(nn_model, UnifiedTransformer):
        ln_out = nn_model.ln_final(h)
        return nn_model.unembed(ln_out)
    else:
        ln_out = nn_model.model.norm(h)
        return nn_model.lm_head(ln_out)


def get_next_token_probs(nn_model: NNLanguageModel) -> LanguageModelProxy:
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
    nn_model: NNLanguageModel,
    prompts,
    layers=None,
    get_activations: GetModuleOutput | None = None,
    remote=False,
    idx=None,
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
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index but it works for both left and right padding
    last_token_index = tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1)
    if idx is None:
        idx = last_token_index.sub(1)  # Default to the last token
    elif idx < 0:
        idx = last_token_index + idx
    else:
        raise ValueError(
            "positive index is currently not supported due to left padding"
        )
    if layers is None:
        layers = range(get_num_layers(nn_model))

    def wrap(h):
        if open_context:
            return h.cpu().save()
        return h

    # Collect the hidden states of the last token of each prompt at each layer
    context = nn_model.trace(prompts, remote=remote) if open_context else nullcontext()
    with context:
        acts = [
            wrap(
                get_activations(nn_model, layer)[
                    th.arange(len(tok_prompts.input_ids)),
                    idx,
                ]
            )
            for layer in layers
        ]
    return th.stack(acts)


def collect_activations_batched(
    nn_model: NNLanguageModel,
    prompts,
    batch_size,
    layers=None,
    get_activations: GetModuleOutput | None = None,
    remote=False,
    idx=None,
    tqdm=None,
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
        idx: The index of the token to collect activations for

    Returns:
        The hidden states of the last token of each prompt at each layer, moved to cpu. Dimensions are (num_layers, num_prompts, hidden_size)
    """
    dataloader = DataLoader(prompts, batch_size=batch_size)
    if tqdm is not None:
        dataloader = tqdm(dataloader)
    acts = []
    for batch in dataloader:
        acts_batch = collect_activations(
            nn_model, batch, layers, get_activations, remote, idx
        )
        acts.append(acts_batch)
    return th.cat(acts, dim=1)
