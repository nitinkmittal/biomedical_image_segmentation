from sklearn.metrics import rand_score
import torch

def pixel_error(
    target: torch.Tensor, 
    pred: torch.Tensor) -> float:
    """Compute pixel error for single image."""
    return (target != pred).sum() / torch.numel(target)

def rand_error(
    target: torch.Tensor, 
    pred: torch.Tensor):
    """Compute rand error for single image."""
    return 1 - rand_score(target.flatten(), pred.flatten())