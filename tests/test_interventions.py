import pytest
import torch as th
from nnterp.nnsight_utils import load_model, get_num_layers
from nnterp.interventions import (
    logit_lens,
    TargetPrompt,
    TargetPromptBatch,
    patchscope_lens,
    steer,
    LatentPrompt,
    run_latent_prompt,
)


@pytest.fixture(
    params=["Maykeye/TinyLLama-v0", "gpt2", "bigscience/bigscience-small-testing"]
)
def model_name(request):
    return request.param


@pytest.fixture
def model(model_name):
    return load_model(model_name, use_module_renaming=True)


def test_logit_lens(model):
    """Test logit lens returns correct dimensions and doesn't crash."""
    prompts = ["Hello world", "Test prompt"]

    result = logit_lens(model, prompts)

    # Should return (num_prompts, num_layers, vocab_size)
    expected_shape = (len(prompts), get_num_layers(model), model.config.vocab_size)
    assert result.shape == expected_shape


def test_logit_lens_single_prompt(model):
    """Test logit lens with single prompt."""
    prompt = "Hello world"

    result = logit_lens(model, prompt)

    # Should return (1, num_layers, vocab_size) for single prompt
    expected_shape = (1, get_num_layers(model), model.config.vocab_size)
    assert result.shape == expected_shape


def test_target_prompt_batch():
    """Test TargetPromptBatch functionality."""
    prompts = [TargetPrompt("Hello", -1), TargetPrompt("World", 0)]
    batch = TargetPromptBatch.from_target_prompts(prompts)

    assert len(batch) == 2
    assert batch.prompts == ["Hello", "World"]
    assert th.equal(batch.index_to_patch, th.tensor([-1, 0]))


def test_patchscope_lens(model):
    """Test patchscope lens with default repeat prompt."""
    source_prompts = ["Hello world", "Test prompt"]

    result = patchscope_lens(model, source_prompts)

    # Should return (num_prompts, num_layers, vocab_size)
    expected_shape = (
        len(source_prompts),
        get_num_layers(model),
        model.config.vocab_size,
    )
    assert result.shape == expected_shape


def test_patchscope_lens_custom_target(model):
    """Test patchscope lens with custom target prompt."""
    source_prompts = ["Hello world"]
    target_prompt = TargetPrompt("The answer is ?", -1)

    result = patchscope_lens(model, source_prompts, target_prompt)

    expected_shape = (1, get_num_layers(model), model.config.vocab_size)
    assert result.shape == expected_shape


def test_steer(model):
    """Test steering intervention doesn't crash."""
    prompt = "Hello world"
    steering_vector = th.randn(model.config.hidden_size)

    # Test that we can run steering without errors
    with model.trace(prompt):
        steer(model, layers=0, steering_vector=steering_vector, factor=1.0)
        output = model.lm_head.output.save()

    # Just check it returns something reasonable
    assert output.shape[-1] == model.config.vocab_size


def test_latent_prompt_creation():
    """Test LatentPrompt creation and methods."""
    prompt = LatentPrompt("Hello ? world", [-1])

    assert prompt.prompt == "Hello ? world"
    assert prompt.latent_spots == [-1]


def test_run_latent_prompt(model):
    """Test run_latent_prompt with actual prompts."""
    latent_prompt = LatentPrompt("The answer is ?", [-1])
    source_prompts = ["Hello world"]

    result = run_latent_prompt(
        model,
        latent_prompt,
        prompts=source_prompts,
        patch_from_layer=0,
        patch_until_layer=min(2, get_num_layers(model) - 1),
    )

    # Should return probabilities for next token
    assert result.shape == (
        1,
        model.config.vocab_size,
    )


def test_interventions_with_multiple_layers(model):
    """Test that interventions work across multiple layers."""
    prompts = ["Hello", "World"]
    num_layers = get_num_layers(model)

    # Test logit lens across all layers
    result = logit_lens(model, prompts)
    assert result.shape[1] == num_layers

    # Test patchscope with specific layers
    layers_to_test = [0, num_layers // 2, num_layers - 1]
    result = patchscope_lens(model, prompts, layers=layers_to_test)
    assert result.shape[1] == len(layers_to_test)


def test_target_prompt_batch_auto():
    """Test TargetPromptBatch auto method."""
    prompt = TargetPrompt("Hello", -1)
    batch = TargetPromptBatch.auto(prompt, batch_size=3)

    assert len(batch) == 3
    assert all(p == "Hello" for p in batch.prompts)
