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
from nnterp.rename_utils import get_ignores, mlp_returns_tuple
from transformers import OPTForCausalLM


def get_layer_test(model_name, model, renamed, i):
    if renamed:
        return model.model.layers[i]
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
            pytest.skip(f"Model {model_name} manual test not implemented")


def get_norm(model_name, model, renamed):
    if renamed:
        return model.model.norm
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
            pytest.skip(f"Model {model_name} manual test not implemented")


def get_num_layers_test(model_name, model, renamed):
    if renamed or model_name == "Maykeye/TinyLLama-v0":
        return len(model.model.layers)
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return len(model.transformer.h)
    elif isinstance(model._model, OPTForCausalLM):
        return len(model.model.decoder.layers)
    else:
        try:
            num_layers = len(model.model.layers)
            return num_layers
        except AttributeError:
            pytest.skip(f"Model {model_name} manual test not implemented")


def test_model_renaming_activations_diff(model_name):
    """
    Test that the activations of the model are the same with and without renaming
    """
    # Load models with and without patching
    model_renamed = StandardizedTransformer(model_name)
    model = LanguageModel(model_name, device_map="auto")
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Set up test input
    prompt = "Hello, world!"

    # Function to collect activations
    def get_token_activations(model, renamed=True):
        activations = []
        with model.trace(prompt):
            # Collect layer outputs
            num_layers = get_num_layers_test(model_name, model, renamed)
            for i in range(num_layers):
                layer = get_layer_test(model_name, model, renamed, i)
                activations.append(layer.output[0].save())

            # Collect final layer norm output
            norm = get_norm(model_name, model, renamed)
            activations.append(norm.output[0].save())

            # Collect logits
            activations.append(model.lm_head.output[0].save())

        return activations

    # Collect activations for both models
    activations = get_token_activations(model, renamed=False)
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
    renamed_model.model.norm
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
            if mlp_returns_tuple(model._model):
                mlps_output_direct = mlps_output_direct[0]
            mlps_output_direct = mlps_output_direct.save()
        layer_output_accessor = model.layers_output[0].save()
        layer_output_direct = model.model.layers[0].output[0].save()

        # Test model-level methods
        logits = model.logits.save()
        logits_direct = model.lm_head.output.save()

        # Test token probability methods
        next_probs = model.next_token_probs.save()

        # Test project_on_vocab with layer output
        projected = model.project_on_vocab(layer_output_accessor).save()
        logits_output = model.output.logits.save()

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
    assert logits_output.shape == projected.shape
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
    model = StandardizedTransformer(model_name)
    prompt = "Hello, world!"

    num_layers = get_num_layers(model)
    assert num_layers > 0
    ignores = get_ignores(model._model)
    with model.trace(prompt):
        # Test layer count and access

        # Test layer 0 methods
        _layer = get_layer(model, 0)

        layer_input = get_layer_input(model, 0).save()

        _attention = get_attention(model, 0)

        attn_output = get_attention_output(model, 0).save()
        if "mlp" not in ignores:
            mlps_output = get_mlp_output(model, 0)
            if mlp_returns_tuple(model._model):
                mlps_output = mlps_output[0]
            mlps_output = mlps_output.save()
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
    assert next_probs.shape == (logits.shape[0], logits.shape[2])
    assert logits_output.shape == projected.shape
    if "mlp" not in ignores:
        assert mlps_output.shape == layer_output.shape
    assert attn_output.shape == layer_input.shape

    print("Renamed model methods tested successfully.")
