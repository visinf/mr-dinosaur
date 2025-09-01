"""Implementations of perceptual grouping algorithms.

We denote methods that group input feature together into slots of objects
(either unconditionally) or via additional conditioning signals as perceptual
grouping modules.
"""
from typing import Optional

import torch
from torch import nn

import ocl.typing


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        self.kvq_dim = kvq_dim if kvq_dim is not None else dim
        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5  

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.gru = nn.GRUCell(self.kvq_dim, dim)
        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp
    
    def step(self, slots, k, v, masks=None):

        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1) 
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn  
        attn = attn + self.eps 
        attn = attn / attn.sum(dim=-1, keepdim=True)
        updates = torch.einsum("bjhd,bihj->bihd", v, attn)  
        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))
        slots = slots.reshape(bs, -1, self.dim)  
        if self.ff_mlp:
            slots = self.ff_mlp(slots)
        return slots, attn_before_reweighting.mean(dim=2)  


    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)
        return slots, attn


class SlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""
    def __init__(
        self,
        feature_dim: int,  
        object_dim: int, 
        kvq_dim: Optional[int] = None, 
        n_heads: int = 1,  
        iters: int = 3,  
        eps: float = 1e-8, 
        ff_mlp: Optional[nn.Module] = None, 
        positional_embedding: Optional[nn.Module] = None, 
        use_projection_bias: bool = False,  
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,  
    ):
        """Initialize Slot Attention Grouping.

        Args:
            feature_dim: Dimensionality of features to slot attention (after positional encoding).
            object_dim: Dimensionality of slots.
            kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
                `object_dim` is used.
            n_heads: Number of heads slot attention uses.
            iters: Number of slot attention iterations.
            eps: Epsilon in slot attention.
            ff_mlp: Optional module applied slot-wise after GRU update.
            positional_embedding: Optional module applied to the features before slot attention,
                adding positional encoding.
            use_projection_bias: Whether to use biases in key, value, query projections.
            use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
                performs one more iteration of slot attention that is used for the gradient step
                after `iters` iterations of slot attention without gradients. Faster and more memory
                efficient than the standard version, but can not backpropagate gradients to the
                conditioning input.
            use_empty_slot_for_masked_slots: Replace slots masked with a learnt empty slot vector.
        """
        super().__init__()
        self._object_dim = object_dim 
        self.slot_attention = SlotAttention(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )
        
        self.positional_embedding = positional_embedding
        
        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.positional_embedding:
            feature = self.positional_embedding(feature.features, feature.positions)
        else:
            feature = feature.features
    

        slots, attn = self.slot_attention(feature, conditioning, slot_mask)
        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=attn,is_empty=slot_mask
        )