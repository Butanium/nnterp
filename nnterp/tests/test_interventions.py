import pytest
import torch as th
from nnterp.nnsight_utils import get_num_layers
from nnterp.interventions import (
    logit_lens,
    TargetPrompt,
    TargetPromptBatch,
    patchscope_lens,
    patchscope_generate,
    steer,
    repeat_prompt,
    it_repeat_prompt,
    patch_object_attn_lens,
)


def test_logit_lens(model):
    """Test logit lens returns correct dimensions and doesn't crash."""
    with th.no_grad():
        prompts = ["Hello world", "Test prompt"]

        result = logit_lens(model, prompts)

        # Should return (num_prompts, num_layers, vocab_size)
        expected_shape = (len(prompts), get_num_layers(model), model.vocab_size)
        assert result.shape == expected_shape


def test_logit_lens_single_prompt(model):
    """Test logit lens with single prompt."""
    with th.no_grad():
        prompt = "Hello world"

        result = logit_lens(model, prompt)

        # Should return (1, num_layers, vocab_size) for single prompt
        expected_shape = (1, get_num_layers(model), model.vocab_size)
        assert result.shape == expected_shape


def test_target_prompt_batch():
    """Test TargetPromptBatch functionality."""
    with th.no_grad():
        prompts = [TargetPrompt("Hello", -1), TargetPrompt("World", 0)]
        batch = TargetPromptBatch.from_target_prompts(prompts)

        assert len(batch) == 2
        assert batch.prompts == ["Hello", "World"]
        assert th.equal(batch.index_to_patch, th.tensor([-1, 0]))


def test_patchscope_lens(model):
    """Test patchscope lens with default repeat prompt."""
    with th.no_grad():
        source_prompts = ["Hello world", "Test prompt"]

        result = patchscope_lens(model, source_prompts)

        # Should return (num_prompts, num_layers, vocab_size)
        expected_shape = (
            len(source_prompts),
            get_num_layers(model),
            model.vocab_size,
        )
        assert result.shape == expected_shape


def test_patchscope_lens_custom_target(model):
    """Test patchscope lens with custom target prompt."""
    with th.no_grad():
        source_prompts = ["Hello world"]
        target_prompt = TargetPrompt("The answer is ?", -1)

        result = patchscope_lens(model, source_prompts, target_prompt)

        expected_shape = (1, get_num_layers(model), model.vocab_size)
        assert result.shape == expected_shape


def test_steer(model):
    """Test steering intervention doesn't crash."""
    with th.no_grad():
        prompt = "Hello world"
        steering_vector = th.randn(model.hidden_size)

        # Test that we can run steering without errors
        with model.trace(prompt):
            steer(model, layers=0, steering_vector=steering_vector, factor=1.0)
            output = model.lm_head.output.save()

        # Just check it returns something reasonable
        assert output.shape[-1] == model.vocab_size


def test_interventions_with_multiple_layers(model):
    """Test that interventions work across multiple layers."""
    with th.no_grad():
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
    with th.no_grad():
        prompt = TargetPrompt("Hello", -1)
        batch = TargetPromptBatch.auto(prompt, batch_size=3)

        assert len(batch) == 3
        assert all(p == "Hello" for p in batch.prompts)


def test_repeat_prompt():
    """Test repeat_prompt function with default and custom parameters."""
    with th.no_grad():
        # Test default repeat prompt
        default_prompt = repeat_prompt()
        assert isinstance(default_prompt, TargetPrompt)
        assert default_prompt.index_to_patch == -1
        assert "king king" in default_prompt.prompt
        assert "1135 1135" in default_prompt.prompt
        assert "hello hello" in default_prompt.prompt
        assert default_prompt.prompt.endswith("?")

        # Test custom repeat prompt
        custom_prompt = repeat_prompt(
            words=["cat", "dog"], rel="->", sep=" | ", placeholder="X"
        )
        assert "cat->cat" in custom_prompt.prompt
        assert "dog->dog" in custom_prompt.prompt
        assert " | " in custom_prompt.prompt
        assert custom_prompt.prompt.endswith("X")


def test_it_repeat_prompt(model):
    """Test it_repeat_prompt function for instruction-tuned models."""
    with th.no_grad():
        # Test basic functionality
        if model.tokenizer.chat_template is None:
            pytest.skip("Model does not support chat template")
        use_system_prompt = True
        try:
            model.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "a"},
                    {"role": "user", "content": "b"},
                ],
                tokenize=False,
            )
        except Exception as e:
            use_system_prompt = False

        it_prompt = it_repeat_prompt(
            model.tokenizer, use_system_prompt=use_system_prompt
        )
        assert isinstance(it_prompt, TargetPrompt)
        assert it_prompt.index_to_patch == -1
        assert len(it_prompt.prompt) > 0  # Should generate some prompt

        # Test with custom parameters
        custom_it_prompt = it_repeat_prompt(
            model.tokenizer,
            words=["test", "word"],
            complete_prompt=False,
            add_user_instr=False,
            use_system_prompt=False,
        )
        assert isinstance(custom_it_prompt, TargetPrompt)


def test_target_prompt_batch_from_prompts():
    """Test TargetPromptBatch.from_prompts method."""
    with th.no_grad():
        # Test with string and single index
        batch = TargetPromptBatch.from_prompts("Hello", -1)
        assert len(batch) == 1
        assert batch.prompts[0] == "Hello"
        assert batch.index_to_patch[0] == -1

        # Test with list of prompts and list of indices
        batch = TargetPromptBatch.from_prompts(["Hello", "World"], [-1, 0])
        assert len(batch) == 2
        assert batch.prompts == ["Hello", "World"]
        assert th.equal(batch.index_to_patch, th.tensor([-1, 0]))


def test_target_prompt_batch_from_target_prompt():
    """Test TargetPromptBatch.from_target_prompt method."""
    with th.no_grad():
        target_prompt = TargetPrompt("Test", -1)
        batch = TargetPromptBatch.from_target_prompt(target_prompt, 3)

        assert len(batch) == 3
        assert all(p == "Test" for p in batch.prompts)
        assert th.equal(batch.index_to_patch, th.tensor([-1, -1, -1]))


def test_target_prompt_batch_iteration():
    """Test TargetPromptBatch iteration and indexing."""
    with th.no_grad():
        prompts = [TargetPrompt("Hello", -1), TargetPrompt("World", 0)]
        batch = TargetPromptBatch.from_target_prompts(prompts)

        # Test indexing
        assert batch[0].prompt == "Hello"
        assert batch[0].index_to_patch == -1
        assert batch[1].prompt == "World"
        assert batch[1].index_to_patch == 0

        # Test iteration
        iterated_prompts = list(batch)
        assert len(iterated_prompts) == 2
        assert iterated_prompts[0].prompt == "Hello"
        assert iterated_prompts[1].prompt == "World"


def test_patchscope_generate(model):
    """Test patchscope_generate function."""
    with th.no_grad():
        source_prompts = ["The capital of France"]
        target_prompt = TargetPrompt("Paris is located in", -1)

        # Test with limited layers to avoid memory issues
        num_layers = get_num_layers(model)
        test_layers = [0, min(1, num_layers - 1)] if num_layers > 1 else [0]
        try:
            with model.generate("a", max_new_tokens=1):
                pass
        except Exception as e:
            pytest.skip(f"Model does not support generate with nnsight: {e}")

        result = patchscope_generate(
            model,
            prompts=source_prompts,
            target_patch_prompt=target_prompt,
            max_length=5,  # Short generation
            layers=test_layers,
        )

        # Should return a dictionary with layer keys
        assert isinstance(result, dict)
        assert len(result) == len(test_layers)
        for layer_key in test_layers:
            assert layer_key in result
            # Each generation should be a tensor
            assert isinstance(result[layer_key], th.Tensor)


def test_patch_object_attn_lens(model):
    """Test patch_object_attn_lens function."""
    with th.no_grad():
        # This is a complex function that requires attention mechanisms
        # We'll do a basic test to ensure it doesn't crash
        source_prompts = ["Hello world"]
        target_prompts = ["Test prompt here"]

        result = patch_object_attn_lens(
            model,
            source_prompts=source_prompts,
            target_prompts=target_prompts,
            attn_idx_patch=0,
            num_patches=1,  # Minimal patching
        )

        # If it succeeds, check basic shape properties
        assert isinstance(result, th.Tensor)
        assert result.shape[0] == len(target_prompts)
        assert result.shape[1] == get_num_layers(model)


def test_patchscope_lens_with_latents(model):
    """Test patchscope_lens with pre-computed latents."""
    with th.no_grad():
        # First get some latents
        from nnterp.nnsight_utils import get_token_activations

        source_prompts = ["Hello world"]
        latents = get_token_activations(model, source_prompts)

        # Test with these latents
        target_prompt = TargetPrompt("Test prompt", -1)

        # Limit to first few layers
        test_layers = (
            [0, min(1, get_num_layers(model) - 1)] if get_num_layers(model) > 1 else [0]
        )

        result = patchscope_lens(
            model,
            source_prompts=None,  # Use latents instead
            target_patch_prompts=target_prompt,
            layers=test_layers,
            latents=latents,
        )

        expected_shape = (1, len(test_layers), model.vocab_size)
        assert result.shape == expected_shape
