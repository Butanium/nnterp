import torch
import pytest
from nnterp.nnsight_utils import load_model
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


# Define a fixture for model names
@pytest.fixture(
    params=[
        "gpt2",
        "bigscience/bigscience-small-testing",
        "Maykeye/TinyLLama-v0",
    ]
)
def model_name(request):
    return request.param


def get_layer_test(model_name, model, renamed, i):
    if renamed:
        return model.model.layers[i]
    elif model_name == "gpt2":
        return model.transformer.h[i]
    elif model_name == "bigscience/bigscience-small-testing":
        return model.transformer.h[i]
    elif model_name == "Maykeye/TinyLLama-v0":
        return model.model.layers[i]
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_norm(model_name, model, renamed):
    if renamed:
        return model.model.norm
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return model.transformer.ln_f
    elif model_name == "Maykeye/TinyLLama-v0":
        return model.model.norm
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_num_layers_test(model_name, model, renamed):
    if renamed or model_name == "Maykeye/TinyLLama-v0":
        return len(model.model.layers)
    elif model_name in ["gpt2", "bigscience/bigscience-small-testing"]:
        return len(model.transformer.h)
    else:
        raise ValueError(f"Model {model_name} not supported")


def test_model_renaming_activations_diff(model_name):
    """
    Test that the activations of the model are the same with and without renaming
    """
    # Load models with and without patching
    model_renamed = load_model(model_name, use_module_renaming=True)
    model = load_model(model_name, use_module_renaming=False)

    # Set up test input
    prompt = "Hello, world!"

    # Function to collect activations
    def get_token_activations(model, renamed=True):
        print(renamed)
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
    ), "Number of activation layers mismatch"

    for i, (act_renamed, act) in enumerate(zip(activations_renamed, activations)):
        assert torch.allclose(
            act_renamed, act, atol=1e-5
        ), f"Mismatch in activation layer {i}"

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
        else load_model(model_name, use_module_renaming=True)
    )
    normal_model = load_model(model_name, use_module_renaming=False)

    # Test that key attributes exist and are accessible
    renamed_model.model.layers[0].self_attn
    renamed_model.model.layers[0].mlp
    renamed_model.model.norm
    renamed_model.lm_head

    prompt = "Hello, world!"

    # Compare outputs
    with normal_model.trace(prompt):
        normal_out = normal_model.output.logits.save()

    with renamed_model.trace(prompt):
        renamed_out = renamed_model.output.logits.save()

    assert torch.allclose(normal_out, renamed_out)


def test_standardized_transformer_methods(model_name):
    """
    Test both accessor methods and direct module access of StandardizedTransformer
    """
    model = StandardizedTransformer(model_name)
    prompt = "Hello, world!"

    num_layers = model.num_layers
    assert num_layers > 0
    with model.trace(prompt):
        # Test both accessor and direct module access ways
        
        # === Test accessor way ===
        _layer_accessor = model.layers[0]
        _layer_direct = model.model.layers[0]
        layer_input_accessor = model.layers_input[0].save()
        layer_input_direct = model.model.layers[0].input.save()
        _attention_accessor = model.attentions[0]
        _attention_direct = model.model.layers[0].self_attn
        _mlp_accessor = model.mlps[0]
        _mlp_direct = model.model.layers[0].mlp
        attn_output_accessor = model.attention_output[0].save()
        attn_output_direct = model.model.layers[0].self_attn.output[0].save()
        mlp_output_accessor = model.mlp_output[0].save()
        mlp_output_direct = model.model.layers[0].mlp.output.save()
        layer_output_accessor = model.layers_output[0].save()
        layer_output_direct = model.model.layers[0].output[0].save()

        # Test model-level methods
        logits = model.get_logits().save()
        logits_direct = model.lm_head.output.save()

        # Test token probability methods
        next_probs = model.get_next_token_probs().save()

        # Test project_on_vocab with layer output
        projected = model.project_on_vocab(layer_output_accessor).save()
        logits_output = model.output.logits.save()

    # Verify accessor and direct access give same results
    assert torch.allclose(layer_input_accessor, layer_input_direct), "Layer input mismatch between accessor and direct access"
    assert torch.allclose(attn_output_accessor, attn_output_direct), "Attention output mismatch between accessor and direct access"  
    assert torch.allclose(mlp_output_accessor, mlp_output_direct), "MLP output mismatch between accessor and direct access"
    assert torch.allclose(layer_output_accessor, layer_output_direct), "Layer output mismatch between accessor and direct access"
    assert torch.allclose(logits, logits_direct), "Logits mismatch between get_logits() and direct access"

    # Test shape consistency
    assert next_probs.shape[-1] == model.config.vocab_size
    assert projected.shape[-1] == model.config.vocab_size
    assert layer_input_accessor.shape == layer_output_accessor.shape
    assert logits.shape[-1] == model.config.vocab_size
    assert next_probs.shape == (logits.shape[0], logits.shape[2])
    assert logits_output.shape == projected.shape
    assert mlp_output_accessor.shape == layer_output_accessor.shape
    assert attn_output_accessor.shape == layer_input_accessor.shape

    print("StandardizedTransformer both accessor and direct access methods tested successfully.")


@pytest.mark.parametrize("model_type", ["standardized", "renamed"])
def test_renamed_model_methods(model_name, model_type):
    """
    Test all methods with a renamed model loaded via load_model
    """
    model = (
        StandardizedTransformer(model_name)
        if model_type == "standardized"
        else load_model(model_name, use_module_renaming=True)
    )
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
    assert next_probs.shape == (logits.shape[0], logits.shape[2])
    assert logits_output.shape == projected.shape
    assert mlp_output.shape == layer_output.shape
    assert attn_output.shape == layer_input.shape

    print("Renamed model methods tested successfully.")
