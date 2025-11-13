import torch
import torch.nn as nn

class FocalCE(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.softmax(logits, dim=1)[torch.arange(targets.size(0), device=logits.device), targets]
        loss = (1 - pt).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()
