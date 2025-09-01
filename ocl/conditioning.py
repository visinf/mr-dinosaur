"""Implementation of conditioning approaches for slots."""
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

import ocl.typing


class RandomConditioning(nn.Module):
    """Random conditioning with potentially learnt mean and stddev."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
        logsigma_init: Callable[[torch.Tensor], None] = nn.init.xavier_uniform_,
    ):
        super().__init__()
        self.n_slots = n_slots 
        self.object_dim = object_dim  

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int) -> ocl.typing.ConditioningOutput:
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)