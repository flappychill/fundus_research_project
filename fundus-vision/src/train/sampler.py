import numpy as np
import torch
from collections import Counter
from torch.utils.data import WeightedRandomSampler

def class_balanced_alpha(counts, beta=0.999):
    freq = np.array(counts, dtype=np.float32)
    eff = 1 - np.power(beta, freq)
    alpha = (1 - beta) / np.clip(eff, 1e-8, None)
    alpha = alpha / alpha.sum() * len(freq)
    return torch.tensor(alpha, dtype=torch.float32)

def make_class_aware_sampler(ds):
    targets = [y for _, y in ds.samples]
    cnt = Counter(targets)
    weights = [1.0 / cnt[y] for y in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
