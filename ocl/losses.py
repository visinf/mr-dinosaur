import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import Tensor

def pairwise_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise cosine similarity between two tensors.

    Args:
        tensor1 (torch.Tensor): Tensor of shape [N, C].
        tensor2 (torch.Tensor): Tensor of shape [M, C].

    Returns:
        torch.Tensor: Similarity matrix of shape [N, M].
    """

    # Normalize both tensors to unit length
    tensor1 = F.normalize(tensor1, p=2, dim=1)
    tensor2 = F.normalize(tensor2, p=2, dim=1)

    # Compute cosine similarity as matrix multiplication
    similarity_matrix = torch.mm(tensor1, tensor2.T)

    return similarity_matrix


def bipartite_matching(masks, attention_maps):
    """
    Perform bipartite matching to associate motion masks with attention maps.

    Parameters:
    batch_masks (torch.Tensor): The motion masks, tensors with shape (C, H, W).
    attention_maps (torch.Tensor): The predicted attention maps, shape (K, H, W).

    Returns:
    list of tuple: The matched indices for each batch.
    """
    K, H, W = attention_maps.shape
    C = masks.shape[0]

    matched_indices = []
    cost_matrix = []
    C = masks.shape[0]
    for i in range(C):
        for j in range(K):
            cost = F.binary_cross_entropy(attention_maps[j], masks[i].float(), reduction='none')
            cost_matrix.append(cost.mean().item())
    cost_matrix = torch.tensor(cost_matrix).reshape(C, K)

    if C > K:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy(), maximize=False)
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

    matched_indices.append((row_ind, col_ind))
    return matched_indices

class NLLloss_batch(nn.Module):
    """Reconstruction loss with masking capability."""

    def __init__(
            self,
            alpha: float,
            similarity_threshold: float
    ):

        super().__init__()
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold

    def forward(
            self,
            mfg,
            Wfg,
            slot_features,
            alpha_masks,
            mk,
            instance_maps,
    ) -> Tensor:
        """
        Compute the negative log likelihood (NLL) loss with regularization.

        Args:
            mfg (torch.Tensor): The moving foreground masks with shape (B, 1, H, W).
            Wfg (torch.Tensor): The predicted foreground probability maps with shape (B, 1, H, W).
            mk (torch.Tensor): The moving masks with shape (B, K, H, W)
            slot_features (torch.Tensor): Slot features with shape (B, K_slot, C)
            alpha_masks (torch.Tensor): DINOSAUR output with shape (B, K_slot, H, W)
            instance_maps (torch.Tensor): labmda * alpha_masks, shape (B, K_slot, H, W)

        Returns:
            torch.Tensor: The computed NLL loss.
        """
        B = slot_features.shape[0]
        total_loss = 0.0
        # Wfg_new
        Wfg_new_batched = self.match_predictions(alpha_masks, mk, slot_features, instance_maps)
        # Loop over batch elements and compute loss for each
        for b in range(B):
            mfg_b = mfg[b]
            Wfg_b = Wfg[b]
            loss_b = self.nll_loss_new(
                mfg_b, Wfg_b, self.alpha,Wfg_new_batched[b])
            total_loss += loss_b
        return total_loss / B

    def match_predictions(self, alpha_masks, mk, slot_features, instance_maps):
        # Get batch size
        B = slot_features.shape[0]
        # Init list to store matched and unmatched features
        matched_features = []
        unmatched_features = []
        indexes_unmatched_batch = []
        # Iterate over batches
        for b in range(B):
            # Get masks
            alpha_masks_b = alpha_masks[b]
            mk_b = mk[b]
            sums_b = mk_b.sum(dim=(1, 2))
            valid_indices = (sums_b > 0).nonzero(as_tuple=True)[0]
            mk_b = mk_b[valid_indices]
            sums_b = mk_b.sum(dim=(1, 2))
            if mk_b.shape[0] >= alpha_masks_b.shape[0]:
                sums_b, indices = torch.sort(sums_b, descending=True)
                _,topk_idx= torch.topk(sums_b, alpha_masks_b.shape[0]-1,largest=True)
                mk_b = mk_b[topk_idx]
            slot_features_b = slot_features[b]
            # Perform matching
            matched_indices = bipartite_matching(mk_b, alpha_masks_b)[0][-1]
            # Get matched features
            matched_features.append(slot_features_b[matched_indices])
            # Get unmatched features
            indexes_unmatched = torch.tensor(
                [index for index, feature in enumerate(slot_features_b) if index not in matched_indices],
                device=slot_features_b.device,
            )
            indexes_unmatched_batch.append(indexes_unmatched)
            unmatched_features.append(slot_features_b[indexes_unmatched])
        similarity = pairwise_similarity(torch.cat(matched_features, dim=0), torch.cat(unmatched_features, dim=0)).amax(
            dim=0)
        is_unmatched = similarity < self.similarity_threshold
        index = 0
        Wfg_new_batched = []
        for b in range(B):
            num_unmatched = len(indexes_unmatched_batch[b])
            is_unmatched_ = is_unmatched[index:index+num_unmatched]
            index = index + num_unmatched
            Wfg_new_batched.append(instance_maps[b][indexes_unmatched_batch[b][is_unmatched_]].sum(dim=0))
        return Wfg_new_batched  

    def nll_loss_new(self, mfg, Wfg, alpha, Wfg_new):
        """
        Compute the negative log likelihood (NLL) loss with regularization.

        Args:
            mfg (torch.Tensor): The moving foreground masks with shape (1, H, W).
            Wfg (torch.Tensor): The predicted foreground probability maps with shape (1, H, W).
            alpha (float): The weighting hyper-parameter.
            Wfg_new (torch.Tensor): The new predicted foreground probability maps with shape (1, H, W).
        Returns:
            torch.Tensor: The computed NLL loss.
        """
        _, H, W = mfg.shape
        N = H * W
        # Flatten the tensors
        mfg_flat = mfg.flatten()
        Wfg_flat = Wfg.flatten()
        Wfg_new_flat = Wfg_new.flatten()
        # First term: NLL loss
        term1 = -1.0 / N * torch.sum(mfg_flat * torch.log(Wfg_flat + 1e-6))  # Add epsilon to prevent log(0)
        Ns = torch.sum(mfg_flat == 0).float()  # Number of pixels with no motion information
        # Second term: Regularization term    
        term2 = alpha / Ns * torch.sum(Wfg_new_flat * (mfg_flat == 0).float())

        # Total loss
        loss = term1 + term2
        return loss.mean()


class ReconstructionLoss(nn.Module):
    """Simple reconstruction loss."""

    def __init__(
        self,
        loss_type: str,
        weight: float = 1.0,
        normalize_target: bool = False,
        num:int=0,
        alpha:float=0,
    ):
        """Initialize ReconstructionLoss.

        Args:
            loss_type: One of `mse`, `mse_sum`, `l1`, `cosine_loss`, `cross_entropy_sum`.
            weight: Weight of loss, output is multiplied with this value.
            normalize_target: Normalize target using mean and std of last dimension
                prior to computing output.
        """
        super().__init__()
        self.num = num
        self.alpha = alpha
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "mse_sum":
            # Used for slot_attention and video slot attention.
            self.loss_fn = (
                lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
            )
        elif loss_type == "l1":
            self.loss_name = "l1_loss"
            self.loss_fn = nn.functional.l1_loss
        elif loss_type == "cosine":
            self.loss_name = "cosine_loss"
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        elif loss_type == "cross_entropy_sum":
            # Used for SLATE, average is over the first (batch) dim only.
            self.loss_name = "cross_entropy_sum_loss"
            self.loss_fn = (
                lambda x1, x2: nn.functional.cross_entropy(
                    x1.reshape(-1, x1.shape[-1]), x2.reshape(-1, x2.shape[-1]), reduction="sum"
                )
                / x1.shape[0]
            )
        elif loss_type == "bce":
            self.loss_name = "binary_cross_entropy_loss"
            self.loss_fn = nn.functional.binary_cross_entropy
        elif loss_type == "wbce":
            self.loss_name = "weighted_binary_cross_entropy_loss"
            self.loss_fn = wbce
        else:
            raise ValueError(
                f"Unknown loss {loss_type}. Valid choices are (mse, l1, cosine, cross_entropy)."
            )
        # If weight is callable use it to determine scheduling otherwise use constant value.
        self.weight = weight
        self.normalize_target = normalize_target

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        """Compute reconstruction loss.

        Args:
            input: Prediction / input tensor.
            target: Target tensor.

        Returns:
            The reconstruction loss.
        """

        target=target.float()
        target = target.detach()
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = self.loss_fn(input, target)
            
        return self.weight * loss


def lwBCE(m, W):
    """
    Compute the weighted Binary Cross-Entropy (wBCE) loss.

    Parameters:
    m (torch.Tensor): The motion mask, shape (1, H, W).
    W (torch.Tensor): The predicted attention map, shape (1, H, W).

    Returns:
    torch.Tensor: The computed LwBCE loss.
    """
    _, H, W_width = m.shape
    _, H, W_width = W.shape
    N = H * W_width
    m = m.flatten()
    W = W.flatten()
    r = m.sum() / N  
    epsilon = 1e-6
    W = torch.clamp(W, epsilon, 1 - epsilon)
    # Compute weighted BCE loss
    loss = -(2 - r) * m * torch.log(W) - (1 - m) * torch.log(1 - W)

    loss = loss.mean()
    return loss

def wbce(batch_attention_maps, batch_masks):
    """
    Calculate the total LwBCE loss for a batch.

    Parameters:
    batch_attention_maps (torch.Tensor): The predicted attention maps, shape (B, K, H, W).
    batch_masks (torch.Tensor): The motion masks, tensors with shape (B, C, H, W).

    Returns:
    torch.Tensor: The total LwBCE loss.
    """
    B = len(batch_masks)
    total_loss = torch.tensor(0.0, device=batch_attention_maps.device)  
    for b in range(B):
        masks_b = batch_masks[b]
        attention_maps_b = batch_attention_maps[b]
        # Filter out empty channels
        non_empty_mask_indices = torch.where(masks_b.sum(dim=(1, 2)) > 0)[0] 
        masks_b_nonzero = masks_b[non_empty_mask_indices]   

        # If all masks are empty, skip this batch item
        if masks_b_nonzero.shape[0] == 0:
            continue 
       
        matched_indices = bipartite_matching(masks_b_nonzero, attention_maps_b)

        rows, cols = matched_indices[0]
        unique_cols, _ = np.unique(cols, return_inverse=True)  
        for j in unique_cols:
            mask_group_indices = np.where(cols == j)[0]  
            m_group = masks_b_nonzero[rows[mask_group_indices]]
            merged_mask = torch.max(m_group, dim=0, keepdim=True)[0] 
            W = attention_maps_b[j : j + 1]
            loss = lwBCE(merged_mask, W)  
            total_loss += loss
    valid_batch_count = B - (batch_masks.sum(dim=(1, 2, 3)) == 0).sum().item()
    total_loss /= max(valid_batch_count, 1)  
    return total_loss