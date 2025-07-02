import pytest
import torch as th
from nnsight import LanguageModel
from nnterp.nnsight_utils import (
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
    set_layer_output,
    skip_layers,
    skip_layer,
)


def test_load_model(llama_like_model_name):
    """Test loading model with different configurations"""
    LanguageModel(llama_like_model_name, device_map="auto")


def test_basic_utils(llama_like_model_name):
    """Test basic utility functions"""
    model = LanguageModel(llama_like_model_name, device_map="auto")
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
        mlps_output = get_mlp_output(model, 0).save()
        layer_output = get_layer_output(model, 0).save()
        if num_layers > 1:
            set_layer_output(model, 1, layer_output)
        else:
            set_layer_output(model, 0, layer_output + 2)
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
    assert mlps_output.shape == layer_output.shape
    assert attn_output.shape == layer_input.shape


def test_activation_collection(llama_like_model_name):
    """Test activation collection functions"""
    model = LanguageModel(llama_like_model_name, device_map="auto")
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


def test_skip_layers(llama_like_model_name):
    """Test skip_layers function"""
    model = LanguageModel(llama_like_model_name, device_map="auto")
    prompt = "Hello, world!"

    # Get baseline output without skipping
    with model.trace(prompt):
        baseline_output = model.lm_head.output.save()

    # Test skipping a single layer range
    for r in [1, 2, 3]:
        if get_num_layers(model) >= r:
            break
        with model.trace(prompt):
            skip_layers(model, r - 1, r)
            skip_output = model.lm_head.output.save()

        # Assert that skipping layers changes the output
        assert not th.allclose(
            baseline_output, skip_output
        ), "Skipping layers should change model output"

        if get_num_layers(model) - r < 1:
            break
        # Test skipping larger range (similar to the example)
        with model.trace(prompt):
            skip_layers(model, 1, get_num_layers(model) - r)
            large_skip_output = model.lm_head.output.save()

        # Test equivalence with manual intervention (replicating the example)
        with model.trace(prompt):
            set_layer_output(
                model, get_num_layers(model) - r, get_layer_output(model, 0)
            )
            manual_skip_output = model.lm_head.output.save()

        # The outputs should be similar when manually setting layer 5 to layer 0's output
        # vs skipping layers 1-5 (which effectively does the same thing)
        assert th.allclose(
            large_skip_output, manual_skip_output, atol=1e-5
        ), f"skip_layers should be equivalent to manually setting layer outputs for r={r}"

        # Test that outputs are different from baseline
        assert not th.allclose(
            baseline_output, large_skip_output
        ), "Skipping multiple layers should significantly change output"

    # Test edge case: skipping only the first layer
    with model.trace(prompt):
        skip_layers(model, 0, 0)
        skip_first_layer_output = model.lm_head.output.save()

    # Skipping the first layer should change the output significantly
    assert not th.allclose(
        baseline_output, skip_first_layer_output
    ), "Skipping the first layer should change output significantly"


def test_skip_layer_and_skip_with(llama_like_model_name):
    """Test skip_layer function and skip_with parameter"""
    model = LanguageModel(llama_like_model_name, device_map="auto")
    prompt = "Hello, world!"
    for layer in [0, 1, -1]:
        if layer >= get_num_layers(model):
            break
        # Test skip_layer function (single layer skipping)
        with model.trace(prompt):
            skip_layer(model, layer)
            single_skip_output = model.lm_head.output.save()

        # Test that skip_layer(2) is equivalent to skip_layers(2, 2)
        with model.trace(prompt):
            skip_layers(model, layer, layer)
            equivalent_skip_output = model.lm_head.output.save()

        assert th.allclose(
            single_skip_output, equivalent_skip_output, atol=1e-5
        ), "skip_layer should be equivalent to skip_layers with same start and end layer"

    # Test multiple skip_layer calls vs single skip_layers call
    for r in [1, 2, 3]:
        if get_num_layers(model) - r < 1:
            break
        with model.trace(prompt):
            for i in range(get_num_layers(model) - r):
                skip_layer(model, i)
            multiple_single_skips = model.lm_head.output.save()

        with model.trace(prompt):
            skip_layers(model, 0, get_num_layers(model) - 1 - r)
            single_multiple_skip = model.lm_head.output.save()

        assert th.allclose(
            multiple_single_skips, single_multiple_skip, atol=1e-5
        ), "Multiple skip_layer calls should be equivalent to single skip_layers call"

    # Test skip_with parameter - use layer 0 output as input to skip layer 3

    # Test skip_with parameter in skip_layers
    if get_num_layers(model) > 3:
        with model.trace(prompt):
            layer_0_output = get_layer_output(model, 0)
            skip_layers(model, 1, 3, skip_with=layer_0_output)
            skip_layers_with_custom_output = model.lm_head.output.save()

    # Test that skip_with=None (default) vs explicit layer input are equivalent
    with model.trace(prompt):
        skip_layer(model, 0)  # skip_with=None (default)
        default_skip_output = model.lm_head.output.save()

    with model.trace(prompt):
        layer_0_input = get_layer_input(model, 0)
        skip_layer(model, 0, skip_with=layer_0_input)
        explicit_input_skip_output = model.lm_head.output.save()

    assert th.allclose(
        default_skip_output, explicit_input_skip_output, atol=1e-5
    ), "skip_with=None should be equivalent to skip_with=get_layer_input(layer)"
    if get_num_layers(model) > 3:
        with model.trace(prompt):
            layer_0_output = get_layer_output(model, 0)
            skip_layer(model, 3, skip_with=layer_0_output)
            skip_with_custom_output = model.lm_head.output.save()
        # Verify that custom skip_with produces different results than default
        assert not th.allclose(
            skip_with_custom_output, default_skip_output
        ), "Using custom skip_with should produce different results than default"


def test_skip_all_layers(llama_like_model_name):
    """Test skipping all layers"""
    model = LanguageModel(llama_like_model_name, device_map="auto")
    prompt = "Hello, world!"
    with model.trace(prompt):
        baseline_output = model.lm_head.output.save()
    with model.trace(prompt):
        skip_layers(model, 0, get_num_layers(model) - 1)
        skip_output = model.lm_head.output.save()
    assert not th.allclose(
        baseline_output, skip_output
    ), "Skipping all layers should change model output"


def test_skip_all_layers_but_one(llama_like_model_name):
    """Test skipping all layers but one"""
    model = LanguageModel(llama_like_model_name, device_map="auto")
    prompt = "Hello, world!"
    with model.trace(prompt):
        baseline_output = model.lm_head.output.save()
    with model.trace(prompt):
        skip_layers(model, 0, get_num_layers(model) - 2)
        skip_output = model.lm_head.output.save()
    assert not th.allclose(
        baseline_output, skip_output
    ), "Skipping all layers but one should change model output"
