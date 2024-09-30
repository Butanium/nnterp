import torch
import pytest
from nnterp.nnsight_utils import load_model, collect_activations, collect_activations_batched

def test_collect_activations_consistency():
    remote = False
    # Load the model
    model = load_model("meta-llama/Meta-Llama-3.1-8B", use_tl=False)
    
    # Define test prompts
    prompts = ["Hello, how are you?", "The weather is nice today.", "I love programming!"]
    
    # Collect activations using the three methods
    activations_standard = collect_activations(model, prompts, remote=remote)
    activations_batched_no_session_1 = collect_activations_batched(model, prompts, batch_size=1, use_session=False, remote=remote)
    activations_batched_session_1 = collect_activations_batched(model, prompts, batch_size=1, use_session=True, remote=remote)
    activations_batched_session = collect_activations_batched(model, prompts, batch_size=2, use_session=True, remote=remote)
    activations_batched_no_session = collect_activations_batched(model, prompts, batch_size=2, use_session=False, remote=remote)
    activations_batched_no_session_3 = collect_activations_batched(model, prompts, batch_size=3, use_session=False, remote=remote)
    activations_batched_session_3 = collect_activations_batched(model, prompts, batch_size=3, use_session=True, remote=remote)
    
    print(activations_standard.shape)
    # Assert that all methods return tensors of the same shape
    assert activations_standard.shape == activations_batched_session.shape == activations_batched_no_session.shape
    
    # Function to calculate mean absolute difference
    def mean_abs_diff(a, b):
        return torch.mean(torch.abs(a - b)).item()
    
    # Assert that the activations are close to each other (allowing for small numerical differences)
    print(f"Mean abs diff (standard vs batched_no_session_3): {mean_abs_diff(activations_standard, activations_batched_no_session_3)}")
    torch.testing.assert_close(activations_standard, activations_batched_no_session_3, rtol=1e-5, atol=1e-5)
    
    print(f"Mean abs diff (standard vs batched_session_3): {mean_abs_diff(activations_standard, activations_batched_session_3)}")
    torch.testing.assert_close(activations_standard, activations_batched_session_3, rtol=1e-5, atol=1e-5)

    print(f"Mean abs diff (batched_session_1 vs batched_no_session_1): {mean_abs_diff(activations_batched_session_1, activations_batched_no_session_1)}")
    torch.testing.assert_close(activations_batched_session_1, activations_batched_no_session_1, rtol=1e-5, atol=1e-5)
    
    print(f"Mean abs diff (standard vs batched_no_session_1): {mean_abs_diff(activations_batched_no_session, activations_batched_no_session_1)}")
    torch.testing.assert_close(activations_batched_no_session, activations_batched_no_session_1, rtol=1e-5, atol=1e-5)
    
    print(f"Mean abs diff (standard vs batched_session_1): {mean_abs_diff(activations_standard, activations_batched_session_1)}")
    torch.testing.assert_close(activations_standard, activations_batched_session_1, rtol=1e-5, atol=1e-5)
    
    print(f"Mean abs diff (batched_session vs batched_no_session): {mean_abs_diff(activations_batched_session, activations_batched_no_session)}")
    torch.testing.assert_close(activations_batched_session, activations_batched_no_session, rtol=1e-5, atol=1e-5)
    
    print(f"Mean abs diff (standard vs batched_no_session): {mean_abs_diff(activations_standard, activations_batched_no_session)}")
    torch.testing.assert_close(activations_standard, activations_batched_no_session, rtol=1e-5, atol=1e-5)
    
    print(f"Mean abs diff (standard vs batched_session): {mean_abs_diff(activations_standard, activations_batched_session)}")
    torch.testing.assert_close(activations_standard, activations_batched_session, rtol=1e-5, atol=1e-5)

    print("All activation collection methods produce consistent results.")

if __name__ == "__main__":
    test_collect_activations_consistency()
