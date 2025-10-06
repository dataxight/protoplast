import torch
import numpy as np
import pytest


def loss_fct_original(pred: torch.Tensor, 
            y: torch.Tensor, 
            perts: np.ndarray, 
            loss_weight: torch.Tensor,
            ctrl: torch.Tensor = None, 
            direction_lambda: float = 1e-3, 
            dict_filter: dict = None, 
            use_mse_loss: bool = False):
    """Original implementation"""
    gamma = 2
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        pred_p = pred[pert_idx].mean(dim=1)
        y_p = y[pert_idx].mean(dim=1)
        ctrl_p = ctrl[pert_idx].mean(dim=1)
        weights = loss_weight[pert_idx]
        if not use_mse_loss:
            losses = losses + torch.sum(weights * (pred_p - y_p + 1)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(weights * (pred_p - y_p + 1)**2)/pred_p.shape[0]/pred_p.shape[1]

        if not use_mse_loss:
            losses = losses + torch.sum(weights * direction_lambda *
                                (torch.sign(y_p - ctrl_p) -
                                 torch.sign(pred_p - ctrl_p))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
    return losses/(len(set(perts)))


def loss_fct_vectorized(pred: torch.Tensor, 
            y: torch.Tensor, 
            perts: torch.Tensor,
            loss_weight: torch.Tensor,
            ctrl: torch.Tensor = None, 
            direction_lambda: float = 1e-3, 
            dict_filter: dict = None, 
            use_mse_loss: bool = False):
    """Vectorized implementation"""
    gamma = 2
    device = pred.device
    
    # Handle string perturbations by mapping to integers
    if not isinstance(perts, torch.Tensor):
        if perts.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
            # Map string perturbations to integers
            unique_strs, inverse = np.unique(perts, return_inverse=True)
            perts = torch.from_numpy(inverse).to(device)
        else:
            perts = torch.from_numpy(perts).to(device)
    elif perts.device != device:
        perts = perts.to(device)
    
    pred_mean = pred.mean(dim=1)  # [N, G]
    y_mean = y.mean(dim=1)  # [N, G]
    ctrl_mean = ctrl.mean(dim=1)  # [N, G]
    
    error = pred_mean - y_mean
    
    if not use_mse_loss:
        main_loss = loss_weight * (error + 1) ** (2 + gamma)
    else:
        main_loss = loss_weight * (error + 1) ** 2
    
    if not use_mse_loss:
        sign_diff = torch.sign(y_mean - ctrl_mean) - torch.sign(pred_mean - ctrl_mean)
        dir_loss = loss_weight * direction_lambda * (sign_diff ** 2)
        total_loss_per_sample = main_loss + dir_loss  # [N, G]
    else:
        total_loss_per_sample = main_loss  # [N, G]
    
    unique_perts, inverse_indices = torch.unique(perts, return_inverse=True)
    n_perts = len(unique_perts)
    
    # For each perturbation, we need to:
    # 1. Sum all losses for samples in that perturbation
    # 2. Divide by (num_samples_in_pert * num_genes)
    
    # Sum losses for each perturbation
    loss_per_pert = torch.zeros(n_perts, device=device)
    sample_count_per_pert = torch.zeros(n_perts, device=device)
    
    # Sum the total loss across genes for each sample
    sample_losses = total_loss_per_sample.sum(dim=1)  # [N]
    
    # Scatter add to group by perturbation
    loss_per_pert.scatter_add_(0, inverse_indices, sample_losses)
    sample_count_per_pert.scatter_add_(0, inverse_indices, torch.ones(pred.shape[0], device=device))
    
    # Divide by (num_samples * num_genes) for each perturbation
    num_genes = pred_mean.shape[1]  # This is G, not T!
    avg_loss_per_pert = loss_per_pert / (sample_count_per_pert * num_genes)
    
    # Average across perturbations
    return avg_loss_per_pert.mean()

class TestLossFunctionEquivalence:
    """Test suite to verify vectorized implementation matches original"""
    
    @pytest.fixture
    def setup_data(self):
        """Create test data"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        N = 100  # number of samples
        T = 5    # time/feature dimension
        G = 50   # number of genes
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        pred = torch.randn(N, T, G, device=device)
        y = torch.randn(N, T, G, device=device)
        ctrl = torch.randn(N, T, G, device=device)
        loss_weight = torch.ones(N, G, device=device)
        
        # Create perturbations with repeats
        perts_np = np.random.choice(['pert_a', 'pert_b', 'pert_c', 'pert_d'], size=N)
        
        return pred, y, ctrl, loss_weight, perts_np, device
    
    def test_basic_equivalence_no_mse(self, setup_data):
        """Test basic case without MSE loss"""
        pred, y, ctrl, loss_weight, perts_np, device = setup_data
        
        loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, 
                                      direction_lambda=1e-3, use_mse_loss=False)
        loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, 
                                       direction_lambda=1e-3, use_mse_loss=False)
        
        print(f"Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
        assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
            f"Losses don't match: {loss_orig.item()} vs {loss_vec.item()}"
    
    def test_basic_equivalence_with_mse(self, setup_data):
        """Test basic case with MSE loss"""
        pred, y, ctrl, loss_weight, perts_np, device = setup_data
        
        loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, 
                                      direction_lambda=1e-3, use_mse_loss=True)
        loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, 
                                       direction_lambda=1e-3, use_mse_loss=True)
        
        print(f"Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
        assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
            f"Losses don't match: {loss_orig.item()} vs {loss_vec.item()}"
    
    def test_different_direction_lambda(self, setup_data):
        """Test with different direction lambda values"""
        pred, y, ctrl, loss_weight, perts_np, device = setup_data
        
        for direction_lambda in [0.0, 1e-5, 1e-2, 1.0]:
            loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, 
                                          direction_lambda=direction_lambda, use_mse_loss=False)
            loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, 
                                           direction_lambda=direction_lambda, use_mse_loss=False)
            
            print(f"Lambda={direction_lambda}: Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
            assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
                f"Losses don't match for lambda={direction_lambda}: {loss_orig.item()} vs {loss_vec.item()}"
    
    def test_single_perturbation(self):
        """Test edge case with single perturbation type"""
        torch.manual_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        N, T, G = 50, 5, 30
        pred = torch.randn(N, T, G, device=device)
        y = torch.randn(N, T, G, device=device)
        ctrl = torch.randn(N, T, G, device=device)
        loss_weight = torch.ones(N, G, device=device)
        perts_np = np.array(['pert_a'] * N)
        
        loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        
        print(f"Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
        assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
            f"Losses don't match for single perturbation: {loss_orig.item()} vs {loss_vec.item()}"
    
    def test_many_perturbations(self):
        """Test with many different perturbation types"""
        torch.manual_seed(42)
        np.random.seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        N, T, G = 200, 5, 40
        pred = torch.randn(N, T, G, device=device)
        y = torch.randn(N, T, G, device=device)
        ctrl = torch.randn(N, T, G, device=device)
        loss_weight = torch.ones(N, G, device=device)
        
        # Create 20 different perturbations
        perts_np = np.random.choice([f'pert_{i}' for i in range(20)], size=N)
        
        loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        
        print(f"Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
        assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
            f"Losses don't match for many perturbations: {loss_orig.item()} vs {loss_vec.item()}"
    
    def test_varying_loss_weights(self, setup_data):
        """Test with non-uniform loss weights"""
        pred, y, ctrl, loss_weight, perts_np, device = setup_data
        
        # Create varying weights
        loss_weight = torch.rand_like(loss_weight) * 2
        
        loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        
        print(f"Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
        assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
            f"Losses don't match with varying weights: {loss_orig.item()} vs {loss_vec.item()}"
    
    def test_gradient_equivalence(self, setup_data):
        """Test that gradients are equivalent"""
        pred, y, ctrl, loss_weight, perts_np, device = setup_data
        
        # Enable gradients
        pred_orig = pred.clone().requires_grad_(True)
        pred_vec = pred.clone().requires_grad_(True)
        
        loss_orig = loss_fct_original(pred_orig, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        loss_orig.backward()
        
        loss_vec = loss_fct_vectorized(pred_vec, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        loss_vec.backward()
        
        print(f"Gradient mean diff: {(pred_orig.grad - pred_vec.grad).abs().mean().item()}")
        assert torch.allclose(pred_orig.grad, pred_vec.grad, rtol=1e-4, atol=1e-5), \
            "Gradients don't match between implementations"
    
    def test_numeric_perts(self):
        """Test with numeric perturbation indices instead of strings"""
        torch.manual_seed(42)
        np.random.seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        N, T, G = 100, 5, 50
        pred = torch.randn(N, T, G, device=device)
        y = torch.randn(N, T, G, device=device)
        ctrl = torch.randn(N, T, G, device=device)
        loss_weight = torch.ones(N, G, device=device)
        perts_np = np.random.randint(0, 5, size=N)
        
        loss_orig = loss_fct_original(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        loss_vec = loss_fct_vectorized(pred, y, perts_np, loss_weight, ctrl, use_mse_loss=False)
        
        print(f"Original: {loss_orig.item()}, Vectorized: {loss_vec.item()}")
        assert torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6), \
            f"Losses don't match with numeric perts: {loss_orig.item()} vs {loss_vec.item()}"


if __name__ == "__main__":
    # Run tests without pytest
    test_suite = TestLossFunctionEquivalence()
    
    # Create setup data
    setup_data = test_suite.setup_data()
    
    print("Running unit tests...")
    print("=" * 80)
    
    try:
        test_suite.test_basic_equivalence_no_mse(setup_data)
        print("✓ test_basic_equivalence_no_mse passed\n")
    except AssertionError as e:
        print(f"✗ test_basic_equivalence_no_mse failed: {e}\n")
    
    try:
        test_suite.test_basic_equivalence_with_mse(setup_data)
        print("✓ test_basic_equivalence_with_mse passed\n")
    except AssertionError as e:
        print(f"✗ test_basic_equivalence_with_mse failed: {e}\n")
    
    try:
        test_suite.test_different_direction_lambda(setup_data)
        print("✓ test_different_direction_lambda passed\n")
    except AssertionError as e:
        print(f"✗ test_different_direction_lambda failed: {e}\n")
    
    try:
        test_suite.test_single_perturbation()
        print("✓ test_single_perturbation passed\n")
    except AssertionError as e:
        print(f"✗ test_single_perturbation failed: {e}\n")
    
    try:
        test_suite.test_many_perturbations()
        print("✓ test_many_perturbations passed\n")
    except AssertionError as e:
        print(f"✗ test_many_perturbations failed: {e}\n")
    
    try:
        test_suite.test_varying_loss_weights(setup_data)
        print("✓ test_varying_loss_weights passed\n")
    except AssertionError as e:
        print(f"✗ test_varying_loss_weights failed: {e}\n")
    
    try:
        test_suite.test_gradient_equivalence(setup_data)
        print("✓ test_gradient_equivalence passed\n")
    except AssertionError as e:
        print(f"✗ test_gradient_equivalence failed: {e}\n")
    
    try:
        test_suite.test_numeric_perts()
        print("✓ test_numeric_perts passed\n")
    except AssertionError as e:
        print(f"✗ test_numeric_perts failed: {e}\n")
    
    print("=" * 80)
    print("All tests completed!")
