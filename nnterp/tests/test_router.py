import torch as th
import pytest
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenamingError, RenameConfig
from nnterp.tests.utils import TEST_MODELS


def get_moe_models():
    """Get list of MoE models for testing."""
    moe_models = [
        "yujiepan/mixtral-8xtiny-random",
        "yujiepan/qwen1.5-moe-tiny-random", 
        "yujiepan/qwen3-moe-tiny-random",
    ]
    # Filter to only include models that are in TEST_MODELS
    return [model for model in moe_models if model in TEST_MODELS]


def get_non_moe_models():
    """Get list of non-MoE models for testing."""
    non_moe_models = [
        "gpt2",
        "bigscience/bigscience-small-testing",
        "yujiepan/opt-tiny-2layers-random",
    ]
    return [model for model in non_moe_models if model in TEST_MODELS]


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_detection_moe_models(model_name):
    """Test that router components are properly detected in MoE models."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be enabled for MoE models
        assert model.routers_available, f"Router should be available for MoE model {model_name}"
        assert model._router_attr_name is not None, f"Router attribute name should be detected for {model_name}"
        
        # Should be able to access router components
        with model.trace("Hello world"):
            # Find a layer with a router (may not be layer 0 for mixed architectures)
            router_layer = None
            for layer_idx in range(min(8, model.num_layers)):
                try:
                    router = model.routers[layer_idx]
                    if router is not None:
                        router_layer = layer_idx
                        break
                except:
                    continue
            
            assert router_layer is not None, f"Should find at least one router layer in {model_name}"


@pytest.mark.parametrize("model_name", get_non_moe_models())
def test_router_detection_non_moe_models(model_name):
    """Test that router components are properly disabled for non-MoE models."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        # Router should be disabled for non-MoE models
        assert not model.routers_available, f"Router should not be available for non-MoE model {model_name}"
        assert model._router_attr_name is None, f"Router attribute name should be None for non-MoE model {model_name}"
        
        # Router accessors should be None
        assert model.routers is None, "Router accessor should be None for non-MoE models"
        assert model.routers_input is None, "Router input accessor should be None for non-MoE models"
        assert model.routers_output is None, "Router output accessor should be None for non-MoE models"
        assert model.router_probabilities is None, "Router probabilities accessor should be None for non-MoE models"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_module_access(model_name):
    """Test accessing router modules and their weights."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Find a layer with a router
            router_layer = None
            for layer_idx in range(min(8, model.num_layers)):
                try:
                    router = model.routers[layer_idx]
                    if router is not None and hasattr(router, 'weight'):
                        router_layer = layer_idx
                        break
                except:
                    continue
            
            if router_layer is None:
                pytest.skip(f"No router layer found for {model_name}")
            
            # Test router module access
            router = model.routers[router_layer]
            assert router is not None, "Router module should be accessible"
            assert hasattr(router, 'weight'), "Router should have weight attribute"
            
            # Test router weights
            router_weights = router.weight
            assert isinstance(router_weights, th.Tensor), "Router weights should be a tensor"
            assert router_weights.ndim >= 2, "Router weights should be at least 2D"
            
            # Test that weights have reasonable shape (hidden_size x num_experts)
            assert router_weights.shape[0] > 0, "Router weights should have positive first dimension"
            assert router_weights.shape[1] > 0, "Router weights should have positive second dimension"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_io_access(model_name):
    """Test accessing router inputs and outputs."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Find a layer with a router
            router_layer = None
            for layer_idx in range(min(8, model.num_layers)):
                try:
                    router_input = model.routers_input[layer_idx]
                    router_output = model.routers_output[layer_idx]
                    if router_input is not None and router_output is not None:
                        router_layer = layer_idx
                        break
                except:
                    continue
            
            if router_layer is None:
                pytest.skip(f"No router layer found for {model_name}")
            
            # Test router input/output access
            router_input = model.routers_input[router_layer]
            router_output = model.routers_output[router_layer]
            
            assert isinstance(router_input, th.Tensor), "Router input should be a tensor"
            assert isinstance(router_output, th.Tensor), "Router output should be a tensor"
            
            # Router I/O should have reasonable shapes
            batch_size, seq_len = model.input_size
            assert router_input.shape[0] == batch_size, f"Router input batch size should be {batch_size}"
            assert router_input.shape[1] == seq_len, f"Router input sequence length should be {seq_len}"
            assert router_output.shape[0] == batch_size, f"Router output batch size should be {batch_size}"
            assert router_output.shape[1] == seq_len, f"Router output sequence length should be {seq_len}"
            assert router_output.shape[2] > 0, "Router outputs should have positive number of experts"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_probabilities_and_top_k(model_name):
    """Test accessing router probabilities and top_k parameter."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Find a layer with a router
            router_layer = None
            for layer_idx in range(min(8, model.num_layers)):
                try:
                    router_probs = model.router_probabilities[layer_idx]
                    if router_probs is not None:
                        router_layer = layer_idx
                        break
                except:
                    continue
            
            if router_layer is None:
                pytest.skip(f"No router layer found for {model_name}")
            
            # Test router probabilities access
            router_probs = model.router_probabilities[router_layer]
            assert isinstance(router_probs, th.Tensor), "Router probabilities should be a tensor"
            
            # Router probabilities should have reasonable shape
            batch_size, seq_len = model.input_size
            assert router_probs.shape[0] == batch_size, f"Router probs batch size should be {batch_size}"
            assert router_probs.shape[1] == seq_len, f"Router probs sequence length should be {seq_len}"
            assert router_probs.shape[2] > 0, "Router probs should have positive number of experts"
        
        # Test getting top_k parameter
        top_k = model.router_probabilities.get_top_k()
        assert isinstance(top_k, int), "top_k should be an integer"
        assert top_k > 0, "top_k should be positive"
        assert top_k <= 8, "top_k should be reasonable (≤ 8 for typical MoE models)"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_structure_validation(model_name):
    """Test router structure validation."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        # Structure validation should not raise errors
        try:
            model.router_probabilities.check_router_structure()
        except RenamingError as e:
            pytest.fail(f"Router structure validation failed: {e}")


def test_router_custom_naming():
    """Test router with custom naming configuration."""
    # Test with custom router name
    custom_config = RenameConfig(router_name="custom_gate")
    
    # This test is more conceptual since we don't have models with custom naming
    # But we can test that the configuration is accepted
    with th.no_grad():
        model = StandardizedTransformer("gpt2", rename_config=custom_config)
        # Should not crash, router should be disabled for non-MoE model
        assert not model.routers_available


def test_router_ignore_configuration():
    """Test router with ignore configuration."""
    # Test with router ignored
    ignore_config = RenameConfig(ignore_router=True)
    
    moe_models = get_moe_models()
    if not moe_models:
        pytest.skip("No MoE models available for testing")
    
    with th.no_grad():
        model = StandardizedTransformer(moe_models[0], rename_config=ignore_config)
        # Router should be disabled even for MoE model when ignored
        assert not model.routers_available
        assert model.routers is None
        assert model.routers_input is None
        assert model.routers_output is None
        assert model.router_probabilities is None


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_multiple_layers(model_name):
    """Test router access across multiple layers, handling mixed architectures."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            # Test accessing routers in multiple layers (some may not have routers)
            router_layers_found = 0
            for layer_idx in range(min(8, model.num_layers)):  # Test first 8 layers
                try:
                    router = model.routers[layer_idx]
                    if router is not None and hasattr(router, 'weight'):
                        router_layers_found += 1
                        
                        # Test router input/output access
                        router_input = model.routers_input[layer_idx]
                        router_output = model.routers_output[layer_idx]
                        router_probs = model.router_probabilities[layer_idx]
                        
                        assert isinstance(router_input, th.Tensor), f"Router input should be tensor at layer {layer_idx}"
                        assert isinstance(router_output, th.Tensor), f"Router output should be tensor at layer {layer_idx}"
                        assert isinstance(router_probs, th.Tensor), f"Router probabilities should be tensor at layer {layer_idx}"
                except:
                    # This layer doesn't have a router (expected for mixed architectures)
                    continue
            
            # Should find at least one router layer
            assert router_layers_found > 0, f"Should find at least one router layer in {model_name}"


@pytest.mark.parametrize("model_name", get_moe_models())
def test_router_tensor_shapes_consistency(model_name):
    """Test that router tensor shapes are consistent and reasonable."""
    with th.no_grad():
        model = StandardizedTransformer(model_name)
        
        if not model.routers_available:
            pytest.skip(f"Router not available for {model_name}")
        
        with model.trace("Hello world"):
            batch_size, seq_len = model.input_size
            
            # Find a layer with a router
            router_layer = None
            for layer_idx in range(min(8, model.num_layers)):
                try:
                    router = model.routers[layer_idx]
                    if router is not None and hasattr(router, 'weight'):
                        router_layer = layer_idx
                        break
                except:
                    continue
            
            if router_layer is None:
                pytest.skip(f"No router layer found for {model_name}")
            
            # Get router components
            router = model.routers[router_layer]
            router_input = model.routers_input[router_layer]
            router_output = model.routers_output[router_layer]
            router_probs = model.router_probabilities[router_layer]
            top_k = model.router_probabilities.get_top_k()
            
            # Validate shapes
            router_weights = router.weight
            assert router_weights.ndim == 2, "Router weights should be 2D"
            hidden_size, num_experts = router_weights.shape
            
            assert router_input.shape[0] == batch_size, f"Router input batch size should be {batch_size}"
            assert router_input.shape[1] == seq_len, f"Router input seq len should be {seq_len}"
            
            assert router_output.shape == (batch_size, seq_len, num_experts), \
                f"Router outputs shape {router_output.shape} should be ({batch_size}, {seq_len}, {num_experts})"
            
            assert router_probs.shape == router_output.shape, \
                f"Router probabilities shape should match output shape"
            
            # top_k should be reasonable relative to num_experts
            assert top_k <= num_experts, f"top_k ({top_k}) should not exceed num_experts ({num_experts})"
            
            # Hidden size should match model's hidden size if available
            if model.hidden_size is not None:
                assert hidden_size == model.hidden_size, \
                    f"Router input size ({hidden_size}) should match model hidden size ({model.hidden_size})"


def test_router_error_handling():
    """Test error handling for router access."""
    with th.no_grad():
        # Test with non-MoE model
        model = StandardizedTransformer("gpt2")
        
        # Should raise appropriate errors
        with pytest.raises(RenamingError, match="Router access is disabled"):
            model.routers.get_top_k()
        
        with pytest.raises(RenamingError, match="Router access is disabled"):
            model.routers.check_router_structure()
        
        with pytest.raises(RenamingError, match="Router access is disabled"):
            with model.trace("Hello world"):
                _ = model.routers[0]
