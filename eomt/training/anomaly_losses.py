import torch
import torch.nn as nn
import torch.nn.functional as F

def logit_normalize(logits: torch.Tensor, tau: float = 0.04) -> torch.Tensor:

    """Normalize logits by their L2 norm, scaled by tau.

    Args:
        logits: tensor of shape (..., C). The normalization is made on the last
            dimension (the one of the classes).
            tau: temperature.

    Returns:
        Tensor with same shape, with L2 norm across the last dimension equal to
        1 / tau.
    """

    norm = logits.norm(p=2, dim=-1, keepdim=True) + 1e-7 
    return logits / (norm * tau)