import torch

def MSELoss(
    weights: torch.Tensor, 
    pred: torch.Tensor, 
    target: torch.Tensor) -> torch.tensor:
    """
    Loss function applicable for binary classification.
    
    Parameters
    -----------
    weights: 
        weights should consist values between 0 and 1.
        Sum of values in weights equal to 1.
        
    """
    weights = torch.where(target < 1., weights[0], weights[1])

    assert weights.shape == target.shape
    
    return (weights * ((target - pred)**2)).sum()/ weights.sum()    


def BCELoss(
    weights: torch.Tensor, 
    pred: torch.Tensor, 
    target: torch.Tensor, ) -> torch.tensor:
    """
    Loss function applicable for binary classification.
    
    Parameters
    -----------
    weights: 
        weights should consist values between 0 and 1.
        Sum of values in weights equal to 1.
        
    """
    pred = torch.sigmoid(pred)
    
    weights = torch.where(target < 1., weights[0], weights[1])

    assert weights.shape == target.shape
    
    return - (weights * (
        target * torch.log(pred + 1e-9) + (1-target) * torch.log(1-pred + 1e-9))).sum() / weights.sum()   