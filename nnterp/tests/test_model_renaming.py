import torch as th
import pytest
from nnsight import LanguageModel
from nnterp import StandardizedTransformer
from nnterp.nnsight_utils import (
    get_layer,
    get_layer_input,
    get_layer_output,
    get_attention,
    get_attention_output,
    get_logits,
    get_next_token_probs,
    project_on_vocab,
    get_mlp_output,
    get_num_layers,
)
from nnterp.rename_utils import get_ignores
from transformers import OPTForCausalLM
from contextlib import nullcontext


def get_layer_test(model_name, model, renamed, i):
    if renamed:
        return model.layers[i]
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return model.transformer.h[i]
    elif model_name == "Maykeye/TinyLLama-v0":
        return model.model.layers[i]
    elif isinstance(model._model, OPTForCausalLM):
        return model.model.decoder.layers[i]
    else:
        try:
            layer = model.model.layers[i]
            return layer
        except AttributeError:
            return None


def get_norm_test(model_name, model, renamed):
    if renamed:
        return model.ln_final
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return model.transformer.ln_f
    elif model_name == "Maykeye/TinyLLama-v0":
        return model.model.norm
    elif isinstance(model._model, OPTForCausalLM):
        return model.model.decoder.final_layer_norm
    else:
        try:
            norm = model.model.norm
            return norm
        except AttributeError:
            return None


def get_num_layers_test(model_name, model, renamed):
    if renamed:
        return len(model.layers)
    elif model_name == "Maykeye/TinyLLama-v0":
        return len(model.model.layers)
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return len(model.transformer.h)
    elif isinstance(model._model, OPTForCausalLM):
        return len(model.model.decoder.layers)
    else:
        try:
            num_layers = len(model.model.layers)
        except AttributeError:
            return None
        return num_layers


def test_model_renaming_activations_diff(model_name):
    """
    Test that the activations of the model are the same with and without renaming
    """
    with th.no_grad():
        # Load models with and without patching
        model_renamed = StandardizedTransformer(model_name)
        model = LanguageModel(model_name, device_map="auto")
        device = "cuda" if th.cuda.is_available() else "cpu"

        # Set up test input
        prompt = "Hello, world!"

        # Function to collect activations
        def get_token_activations(model, renamed=True):
            activations = []
            num_layers = get_num_layers_test(model_name, model, renamed)
            if (
                num_layers is None
                or get_layer_test(model_name, model, renamed, 0) is None
                or get_norm_test(model_name, model, renamed) is None
            ):
                with open("bump.log", "a") as f:
                    f.write(f"Model {model_name} manual test not implemented\n")
                return None
            with model.trace(prompt):
                # Collect layer outputs
                for i in range(num_layers):
                    layer = get_layer_test(model_name, model, renamed, i)
                    if layer is None:
                        return None
                    activations.append(layer.output[0].save())

                # Collect final layer norm output
                norm = get_norm_test(model_name, model, renamed)
                if norm is None:
                    return None
                activations.append(norm.output[0].save())

                # Collect logits
                activations.append(model.lm_head.output[0].save())

            return activations

        # Collect activations for both models
        activations = get_token_activations(model, renamed=False)
        if activations is None:
            pytest.skip(f"Model {model_name} manual test not implemented")
        activations_renamed = get_token_activations(model_renamed, renamed=True)

        # Compare activations
        assert len(activations_renamed) == len(
            activations
        ), f"Number of activation layers mismatch: {len(activations_renamed)} != {len(activations)}"

        for i, (act_renamed, act) in enumerate(zip(activations_renamed, activations)):
            assert th.allclose(
                act_renamed.to(device), act.to(device), atol=1e-3
            ), f"Mismatch in activation layer {i}. Max diff: {th.max(th.abs(act_renamed.to(device) - act.to(device)))}"

        print("All activations match between renamed and unrenamed models.")


@pytest.mark.parametrize("model_type", ["standardized", "renamed"])
def test_renaming_forward(model_name, model_type):
    """
    Test that the renaming is correct for both StandardizedTransformer and renamed model
    """
    with th.no_grad():
        # Load models
        renamed_model = (
            StandardizedTransformer(model_name)
            if model_type == "standardized"
            else StandardizedTransformer(model_name)
        )
        normal_model = LanguageModel(model_name, device_map="auto")
        device = "cuda" if th.cuda.is_available() else "cpu"

        # Test that key attributes exist and are accessible
        renamed_model.model.layers[0].self_attn
        renamed_model.attentions[0]
        if not isinstance(renamed_model._model, OPTForCausalLM):
            renamed_model.model.layers[0].mlp
            renamed_model.mlps[0]
        renamed_model.model.ln_final
        renamed_model.lm_head

        prompt = "Hello, world!"

        # Compare outputs
        with normal_model.trace(prompt):
            normal_out = normal_model.output.logits.save()

        with renamed_model.trace(prompt):
            renamed_out = renamed_model.output.logits.save()

        assert th.allclose(normal_out.to(device), renamed_out.to(device), atol=1e-5)


def test_standardized_transformer_methods(model_name):
    """
    Test both accessor methods and direct module access of StandardizedTransformer
    """
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        num_layers = model.num_layers
        assert num_layers > 0
        ignores = get_ignores(model._model)
        with model.trace(prompt):
            # Test both accessor and direct module access ways

            # === Test accessor way ===
            _layer_accessor = model.layers[0]
            _layer_direct = model.model.layers[0]
            layer_input_accessor = model.layers_input[0].save()
            layer_input_direct = model.model.layers[0].input.save()
            _attention_accessor = model.attentions[0]
            _attention_direct = model.model.layers[0].self_attn
            if "mlp" not in ignores:
                _mlp_accessor = model.mlps[0]
                _mlp_direct = model.model.layers[0].mlp
            attn_output_accessor = model.attentions_output[0].save()
            attn_output_direct = model.model.layers[0].self_attn.output[0].save()
            if "mlp" not in ignores:
                mlps_output_accessor = model.mlps_output[0].save()
                mlps_output_direct = model.model.layers[0].mlp.output
                if isinstance(mlps_output_direct, tuple):
                    mlps_output_direct = mlps_output_direct[0]
                mlps_output_direct = mlps_output_direct.save()
            layer_output_accessor = model.layers_output[0].save()
            layer_output_direct = model.model.layers[0].output[0].save()

            # Test token probability methods
            next_probs = model.next_token_probs.save()

            # Test project_on_vocab with layer output
            projected = model.project_on_vocab(layer_output_accessor).save()
            # Test model-level methods
            logits = model.logits.save()
            logits_direct = model.output.logits.save()

        # Verify accessor and direct access give same results
    assert th.allclose(
        layer_input_accessor, layer_input_direct
    ), "Layer input mismatch between accessor and direct access"
    assert th.allclose(
        attn_output_accessor, attn_output_direct
    ), "Attention output mismatch between accessor and direct access"
    if "mlp" not in ignores:
        assert th.allclose(
            mlps_output_accessor, mlps_output_direct
        ), "MLP output mismatch between accessor and direct access"
    assert th.allclose(
        layer_output_accessor, layer_output_direct
    ), "Layer output mismatch between accessor and direct access"
    assert th.allclose(
        logits, logits_direct
    ), "Logits mismatch between logits and direct access"

    # Test shape consistency
    assert next_probs.shape[-1] == model.config.vocab_size
    assert projected.shape[-1] == model.config.vocab_size
    assert layer_input_accessor.shape == layer_output_accessor.shape
    assert logits.shape[-1] == model.config.vocab_size
    assert next_probs.shape == (logits.shape[0], logits.shape[2])
    if "mlp" not in ignores:
        assert mlps_output_accessor.shape == layer_output_accessor.shape
    assert attn_output_accessor.shape == layer_input_accessor.shape

    print(
        "StandardizedTransformer both accessor and direct access methods tested successfully."
    )


def test_renamed_model_methods(model_name):
    """
    Test all methods with a renamed model loaded via StandardizedTransformer
    """
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        num_layers = get_num_layers(model)
        assert num_layers > 0
        ignores = get_ignores(model._model)
        with model.trace(prompt):
            batch_size = model.input_size[0].save()
            seq_len = model.input_size[1].save()
            embed_tokens = model.token_embeddings.save()
            embed_tokens_out = model.embed_tokens.output.save()

            # Test layer 0 methods
            _layer = get_layer(model, 0)

            layer_input = get_layer_input(model, 0).save()

            _attention = get_attention(model, 0)

            attn_output = get_attention_output(model, 0).save()
            if "mlp" not in ignores:
                mlps_output = get_mlp_output(model, 0)
                if isinstance(mlps_output, tuple):
                    mlps_output = mlps_output[0]
                mlps_output = mlps_output.save()
            layer_output = get_layer_output(model, 0).save()

            # Test token probability methods
            next_probs = get_next_token_probs(model).save()

            # Test project_on_vocab with layer output
            projected = project_on_vocab(model, layer_output).save()
            # Test model-level methods
            logits = get_logits(model).save()
            logits_output = model.output.logits.save()

        assert next_probs.shape == (batch_size, model.config.vocab_size)
        assert projected.shape == (batch_size, seq_len, model.config.vocab_size)
        assert embed_tokens.shape == (batch_size, seq_len, model.config.hidden_size)
        assert embed_tokens_out.shape == (batch_size, seq_len, model.config.hidden_size)
        assert th.allclose(embed_tokens, embed_tokens_out)
        assert layer_input.shape == (batch_size, seq_len, model.config.hidden_size)
        assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
        assert logits_output.shape == (batch_size, seq_len, model.config.vocab_size)
        if "mlp" not in ignores:
            assert mlps_output.shape == (batch_size, seq_len, model.config.hidden_size)
        assert attn_output.shape == (batch_size, seq_len, model.config.hidden_size)


def test_standardized_transformer_input_accessors(model_name):
    """Test input accessor methods of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        ignores = get_ignores(model._model)
        with model.trace(prompt):
            # Test input accessors
            layer_input_accessor = model.layers_input[0].save()
            layer_input_direct = model.model.layers[0].input.save()

            attn_input_accessor = model.attentions_input[0].save()
            attn_input_direct = model.model.layers[0].self_attn.input.save()

            if "mlp" not in ignores:
                mlp_input_accessor = model.mlps_input[0].save()
                mlp_input_direct = model.model.layers[0].mlp.input.save()

        # Verify input accessors work correctly
    assert th.allclose(
        layer_input_accessor, layer_input_direct
    ), "Layer input accessor mismatch"
    assert th.allclose(
        attn_input_accessor, attn_input_direct
    ), "Attention input accessor mismatch"
    if "mlp" not in ignores:
        assert th.allclose(
            mlp_input_accessor, mlp_input_direct
        ), "MLP input accessor mismatch"


def test_standardized_transformer_steer_method(model_name):
    """Test steer method of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        # Create a random steering vector
        hidden_size = model.hidden_size
        if hidden_size is None:
            pytest.fail(f"Model {model_name} has no hidden size")

        steering_vector = th.randn(hidden_size) * 0.1  # Small perturbation

    # Test steering single layer
    with model.trace(prompt):
        baseline_output = model.logits.save()

    with model.trace(prompt):
        model.steer(layers=0, steering_vector=steering_vector, factor=1.0)
        steered_output = model.logits.save()

    # Steered output should be different from baseline
    assert not th.allclose(
        baseline_output, steered_output, atol=1e-4
    ), "Steering should change model output"

    # Test steering multiple layers
    with model.trace(prompt):
        model.steer(
            layers=list(range(min(model.num_layers, 2))),
            steering_vector=steering_vector,
            factor=0.5,
        )
        multi_steered_output = model.logits.save()

    assert not th.allclose(
        baseline_output, multi_steered_output, atol=1e-4
    ), "Multi-layer steering should change output"

    # Test position-specific steering
    num_tokens = len(model.tokenizer.encode(prompt))
    assert num_tokens > 1, "Prompt should have multiple tokens for position testing"

    with model.trace(prompt):
        model.steer(layers=0, steering_vector=steering_vector, positions=0)
        pos_steered_output = model.logits.save()

    assert not th.allclose(
        baseline_output, pos_steered_output, atol=1e-4
    ), "Position-specific steering should change output"


def test_standardized_transformer_skip_methods(model_name):
    """Test skip methods of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = "Hello, world!"

        if model.num_layers < 2:
            pytest.skip("Model needs at least 2 layers for skip testing")

    # Test skip_layer method
    with model.trace(prompt):
        baseline_output = model.logits.save()

    with model.trace(prompt):
        model.skip_layer(0)
        skip_output = model.logits.save()

    assert not th.allclose(
        baseline_output, skip_output
    ), "skip_layer should change output"

    # Test skip_layers method
    with model.trace(prompt):
        model.skip_layers(0, 1)
        skip_layers_output = model.logits.save()

    assert not th.allclose(
        baseline_output, skip_layers_output
    ), "skip_layers should change output"

    # Test skip with custom tensor
    with model.trace(prompt):
        custom_tensor = model.layers_input[0]
        model.skip_layer(1, skip_with=custom_tensor)
        custom_skip_output = model.logits.save()

    assert not th.allclose(
        baseline_output, custom_skip_output
    ), "skip with custom tensor should change output"


def test_standardized_transformer_constructor_options(model_name):
    """Test StandardizedTransformer constructor options"""
    with th.no_grad():
        # Test with check_renaming=False
        model_no_check = StandardizedTransformer(model_name, check_renaming=False)
        assert model_no_check.num_layers > 0

        # Test with custom renaming parameters - this should still work even with wrong names
        model_custom = StandardizedTransformer(
            model_name, attn_rename="nonexistent_attn", check_renaming=False
        )
        assert model_custom.num_layers > 0


def test_standardized_transformer_properties(model_name):
    """Test properties of StandardizedTransformer"""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        prompt = ["a", "b b b b b"]
        if model.hidden_size is None:
            pytest.fail(f"Model {model_name} has no hidden size")
        if model.num_heads is None:
            pytest.fail(f"Model {model_name} has no num_heads")

        with model.trace(prompt):
            # Test properties
            unembed_norm = model.ln_final
            assert unembed_norm is not None
            next_token_probs = model.next_token_probs.save()
            logits = model.logits.save()

        # Verify shapes and properties
        assert (
            next_token_probs.shape
            == (logits.shape[0], logits.shape[-1])
            == (
                len(prompt),
                model.config.vocab_size,
            )
        )
        assert th.allclose(
            next_token_probs.sum(dim=-1),
            th.ones(logits.shape[0], device=next_token_probs.device),
            atol=1e-5,
        )
