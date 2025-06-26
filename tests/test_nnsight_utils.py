import pytest
import torch as th
from nnterp.nnsight_utils import (
    load_model,
    get_num_layers,
    get_layer,
    get_layer_input,
    get_layer_output,
    get_attention,
    get_attention_output,
    get_logits,
    project_on_vocab,
    get_next_token_probs,
    get_token_activations,
    collect_last_token_activations_session,
    collect_activations_batched,
    compute_next_token_probs,
    get_mlp_output,
)


@pytest.fixture(
    params=[
        "Maykeye/TinyLLama-v0",
    ]
)
def model_name(request):
    return request.param


def test_load_model(model_name):
    """Test loading model with different configurations"""
    load_model(model_name)


def test_basic_utils(model_name):
    """Test basic utility functions"""
    model = load_model(model_name)
    prompt = "Hello, world!"

    num_layers = get_num_layers(model)
    assert num_layers > 0
    with model.trace(prompt):
        # Test layer count and access

        # Test layer 0 methods
        _layer = get_layer(model, 0)

        layer_input = get_layer_input(model, 0).save()

        _attention = get_attention(model, 0)

        attn_output = get_attention_output(model, 0).save()
        mlp_output = get_mlp_output(model, 0).save()
        layer_output = get_layer_output(model, 0).save()

        # Test model-level methods
        logits = get_logits(model).save()

        # Test token probability methods
        next_probs = get_next_token_probs(model).save()

        # Test project_on_vocab with layer output
        projected = project_on_vocab(model, layer_output).save()
        logits_output = model.output.logits.save()

    assert next_probs.shape[-1] == model.config.vocab_size
    assert projected.shape[-1] == model.config.vocab_size
    assert layer_input.shape == layer_output.shape
    assert logits.shape[-1] == model.config.vocab_size
    assert next_probs.shape == (logits.shape[0], model.config.vocab_size)
    assert logits_output.shape == projected.shape
    assert mlp_output.shape == layer_output.shape
    assert attn_output.shape == layer_input.shape


def test_activation_collection(model_name):
    """Test activation collection functions"""
    model = load_model(model_name)
    prompts = ["Hello, world!", "Testing, 1, 2, 3"] * 2

    # Test activation collection with session
    acts_session = collect_last_token_activations_session(model, prompts, batch_size=1)
    assert acts_session.shape[:2] == (get_num_layers(model), len(prompts))
    # Test basic activation collection
    acts = get_token_activations(model, prompts)
    assert acts.shape[:2] == (get_num_layers(model), len(prompts))  # Batch dimension

    # Test batched activation collection
    acts_batched = collect_activations_batched(model, prompts, batch_size=1)
    assert acts_batched.shape[:2] == (get_num_layers(model), len(prompts))

    acts_batched_no_batch = collect_activations_batched(
        model, prompts, batch_size=len(prompts)
    )
    assert th.allclose(acts, acts_batched_no_batch)

    # Test next token probabilities
    probs = compute_next_token_probs(model, prompts)
    assert probs.shape == (len(prompts), model.config.vocab_size)
