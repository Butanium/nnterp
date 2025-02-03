import torch
import pytest
from nnterp.nnsight_utils import load_model
from nnterp import StandardizedTransformer


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


def get_layer(model_name, model, renamed, i):
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


def get_num_layers(model_name, model, renamed):
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
    def collect_activations(model, renamed=True):
        print(renamed)
        activations = []
        with model.trace(prompt):
            # Collect layer outputs
            num_layers = get_num_layers(model_name, model, renamed)
            for i in range(num_layers):
                layer = get_layer(model_name, model, renamed, i)
                activations.append(layer.output[0].save())

            # Collect final layer norm output
            norm = get_norm(model_name, model, renamed)
            activations.append(norm.output[0].save())

            # Collect logits
            activations.append(model.lm_head.output[0].save())

        return [act.value for act in activations]

    # Collect activations for both models
    activations = collect_activations(model, renamed=False)
    activations_renamed = collect_activations(model_renamed, renamed=True)

    # Compare activations
    assert len(activations_renamed) == len(
        activations
    ), "Number of activation layers mismatch"

    for i, (act_renamed, act) in enumerate(zip(activations_renamed, activations)):
        assert torch.allclose(
            act_renamed, act, atol=1e-5
        ), f"Mismatch in activation layer {i}"

    print("All activations match between renamed and unrenamed models.")


def test_renaming_is_correct(model_name):
    """
    Test that the renaming is correct
    """
    model_renamed = StandardizedTransformer(model_name)
    model_loaded = load_model(model_name, use_module_renaming=True)
    normal_model = load_model(model_name, use_module_renaming=False)
    model_renamed.model.layers[0].self_attn
    model_loaded.model.layers[0].self_attn
    model_renamed.model.layers[0].mlp
    model_loaded.model.layers[0].mlp
    model_renamed.model.norm
    model_loaded.model.norm
    model_renamed.lm_head
    model_loaded.lm_head
    prompt = "Hello, world!"

    with normal_model.trace(prompt):
        normal_out = normal_model.output.logits.save()

    with model_renamed.trace(prompt):
        renamed_out = model_renamed.output.logits.save()

    with model_loaded.trace(prompt):
        loaded_out = model_loaded.output.logits.save()

    assert torch.allclose(normal_out, renamed_out)
    assert torch.allclose(normal_out, loaded_out)


def test_standardized_transformer_methods(model_name):
    """
    Test all methods of StandardizedTransformer
    """
    model = StandardizedTransformer(model_name)
    prompt = "Hello, world!"

    num_layers = model.get_num_layers()
    assert num_layers > 0
    with model.trace(prompt):
        # Test layer count and access

        # Test layer 0 methods
        _layer = model.get_layer(0)

        layer_input = model.get_layer_input(0).save()

        layer_output = model.get_layer_output(0).save()

        _attention = model.get_attention(0)

        attn_output = model.get_attention_output(0).save()
        mlp_output = model.get_mlp_output(0).save()

        # Test model-level methods
        logits = model.get_logits().save()

        # Test token probability methods
        next_probs = model.get_next_token_probs().save()

        # Test project_on_vocab with layer output
        projected = model.project_on_vocab(layer_output).save()
        logits_output = model.output.logits.save()
        # Test stop_at_layer
        model.stop_at_layer(0)

    assert next_probs.shape[-1] == model.config.vocab_size
    assert projected.shape[-1] == model.config.vocab_size
    assert layer_input.shape == layer_output.shape
    assert logits.shape[-1] == model.config.vocab_size
    assert next_probs.shape == (logits.shape[0], logits.shape[2])
    assert logits_output.shape == projected.shape
    assert mlp_output.shape == layer_output.shape
    assert attn_output.shape == layer_input.shape

    print("All StandardizedTransformer methods tested successfully.")
