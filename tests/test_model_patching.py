import torch
import pytest
from nnterp.nnsight_utils import load_model, get_num_layers

@pytest.mark.parametrize("model_name", ["gpt2"])
def test_model_patching_activations(model_name):
    # Load models with and without patching
    model_patched = load_model(model_name, use_module_renaming=True)
    model_unpatched = load_model(model_name, use_module_renaming=False)

    # Set up test input
    prompt = "Hello, world!"
    
    # Function to collect activations
    def collect_activations(model, patched=True):
        print(patched)
        activations = []
        with model.trace(prompt):
            # Collect layer outputs
            num_layers = get_num_layers(model)
            for i in range(num_layers):
                layer = model.model.layers[i] if patched else model.model.h[i]
                activations.append(layer.output[0].save())
            
            # Collect final layer norm output
            norm = model.model.norm if patched else model.model.ln_f
            activations.append(norm.output[0].save())
            
            # Collect logits
            activations.append(model.lm_head.output[0].save())
        
        return [act.value for act in activations]

    # Collect activations for both models
    activations_unpatched = collect_activations(model_unpatched, patched=False)
    activations_patched = collect_activations(model_patched, patched=True)

    # Compare activations
    assert len(activations_patched) == len(activations_unpatched), "Number of activation layers mismatch"

    for i, (act_patched, act_unpatched) in enumerate(zip(activations_patched, activations_unpatched)):
        assert torch.allclose(act_patched, act_unpatched, atol=1e-5), f"Mismatch in activation layer {i}"

    print("All activations match between patched and unpatched models.")