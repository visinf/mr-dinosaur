"""Utility functions related to windows of inputs."""
import torch
import torch.nn.functional as F
from torch import nn
import os
from typing import Tuple, Optional

class JoinWindows(torch.nn.Module):
    """Join individual windows to single output."""

    def __init__(self, n_windows: int, size):
        super().__init__()
        self.n_windows = n_windows
        self.size = size  

    def forward(self, masks: torch.Tensor, keys: str) -> torch.Tensor:
        assert len(masks) == self.n_windows

        keys_split = [key.split("_") for key in keys]
        pad_left_list = [int(elems[1]) for elems in keys_split]
        pad_top_list = [int(elems[2]) for elems in keys_split]
        target_height, target_width = self.size
        n_elems = masks.shape[1]
        n_masks = masks.shape[0] * n_elems
        height, width = masks.shape[2], masks.shape[3]

        full_mask = torch.zeros(n_masks, target_height, target_width, dtype=masks.dtype, device=masks.device)

        for idx, mask in enumerate(masks):
            pad_left = pad_left_list[idx]
            pad_top = pad_top_list[idx]

            x_start = 0 if pad_left >= 0 else -pad_left
            x_end = min(width, target_width - pad_left)

            y_start = 0 if pad_top >= 0 else -pad_top
            y_end = min(height, target_height - pad_top)

            cropped = mask[:, y_start:y_end, x_start:x_end]

            if cropped.shape[-1] <= 0 or cropped.shape[-2] <= 0:
                print(f"[WARNING] Window idx={idx} has no valid overlap; skipping.")
                continue

            final_x = max(pad_left, 0)
            final_y = max(pad_top, 0)

            if final_x >= target_width or final_y >= target_height:
                print(f"[WARNING] Window idx={idx} is entirely out of range; skipping.")
                continue

            cropped_h = cropped.shape[-2]
            cropped_w = cropped.shape[-1]

            if final_x + cropped_w > target_width or final_y + cropped_h > target_height:
                print(f"[WARNING] Window idx={idx} partially out of range; skipping or do partial clip.")
                continue

            chan_start = idx * n_elems
            chan_end   = (idx + 1) * n_elems
            full_mask[chan_start:chan_end,
                      final_y:final_y+cropped_h,
                      final_x:final_x+cropped_w] = cropped
        return full_mask.unsqueeze(0)

class DropBatchDims(torch.nn.Module):
    def __init__(self, dim_keep_start: int, dim_keep_end: int):
        super().__init__()
        self.dim_start = dim_keep_start
        self.dim_end = dim_keep_end

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp[self.dim_start : self.dim_end]

class MergeWindows(nn.Module):
    """Join individual windows to single output."""
    def __init__(self, n_windows: int, size: Tuple[int, int], similarity: float, output_dir: Optional[str] = None):
        super().__init__()
        self.n_windows = n_windows
        self.size = size 
        self.output_dir = output_dir
        self.similarity = similarity
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        
    @staticmethod
    def to_one_hot(masks: torch.Tensor) -> torch.Tensor:

        cluster_ids = masks.argmax(dim=1) 
        n_clusters = masks.shape[1]
        masks_oh = F.one_hot(cluster_ids, n_clusters) 
        masks_oh = masks_oh.permute(0, 3, 1, 2).float() 
        return masks_oh
    
    def _parse_keys(self, keys):
        keys_split = [key.split("_") for key in keys]
        pad_left = [int(elems[1]) for elems in keys_split]
        pad_top = [int(elems[2]) for elems in keys_split]
        return pad_left, pad_top
    
    def _get_window_indices(self, n_masks):
       
        masks_per_window = n_masks // self.n_windows
        window_mask_indices = []
        for i in range(self.n_windows):
            start = i * masks_per_window
            end = start + masks_per_window
            window_mask_indices.append((start, end))
        return window_mask_indices
    
    def _find_adjacent_windows(self, pad_left, pad_top):
        n = self.n_windows
        adjacency = []
        for i in range(n):
            for j in range(i+1, n):
                if pad_top[i] == pad_top[j] and abs(pad_left[i] - pad_left[j]) == self.size[1]:
                    if pad_left[i] < pad_left[j]:
                        adjacency.append((i, j, 'horizontal'))
                    else:
                        adjacency.append((j, i, 'horizontal'))
                if pad_left[i] == pad_left[j] and abs(pad_top[i] - pad_top[j]) == self.size[0]:
                    if pad_top[i] < pad_top[j]:
                        adjacency.append((i, j, 'vertical'))
                    else:
                        adjacency.append((j, i, 'vertical'))
        return adjacency
    def _get_edge_masks(self, masks, window_idx, pad_left, pad_top, window_mask_idx_range):

        start_idx, end_idx = window_mask_idx_range
        h, w = self.size
        pl = pad_left[window_idx]
        pt = pad_top[window_idx]

        H = masks.shape[2]
        W = masks.shape[3]

        pt_start = max(pt, 0)
        pt_end = min(pt + h, H)
        pl_start = max(pl, 0)
        pl_end = min(pl + w, W)

        if pt_start >= pt_end or pl_start >= pl_end:
            return {
                'left': set(),
                'right': set(),
                'top': set(),
                'bottom': set()
            }

        window_masks = masks[:, start_idx:end_idx, pt_start:pt_end, pl_start:pl_end]

        edge_mask_info = {
            'left': set(),
            'right': set(),
            'top': set(),
            'bottom': set()
        }

        hh = window_masks.shape[2]
        ww = window_masks.shape[3]

        if ww == 0 or hh == 0:
            return edge_mask_info

        left_edge = window_masks[..., :, 0]       if ww > 0 else None
        right_edge = window_masks[..., :, -1]     if ww > 0 else None
        top_edge = window_masks[..., 0, :]        if hh > 0 else None
        bottom_edge = window_masks[..., -1, :]    if hh > 0 else None

        for c in range(start_idx+1, end_idx):
            c_rel = c - start_idx
            if left_edge is not None and (left_edge[0, c_rel] == 1).any():
                edge_mask_info['left'].add(c)
            if right_edge is not None and (right_edge[0, c_rel] == 1).any():
                edge_mask_info['right'].add(c)
            if top_edge is not None and (top_edge[0, c_rel] == 1).any():
                edge_mask_info['top'].add(c)
            if bottom_edge is not None and (bottom_edge[0, c_rel] == 1).any():
                edge_mask_info['bottom'].add(c)

        return edge_mask_info

    
    def _cosine_similarity(self, f1, f2):

        f1 = f1 / (f1.norm(p=2) + 1e-8)
        f2 = f2 / (f2.norm(p=2) + 1e-8)
        return (f1 * f2).sum()
    
    def _merge_across_boundaries(self, masks, slot_features, pad_left, pad_top, adjacency, window_mask_indices):

        n_masks = masks.shape[1]
        masks_per_window = n_masks // self.n_windows
        
        edge_info_per_window = []
        for wi in range(self.n_windows):
            edge_info = self._get_edge_masks(masks, wi, pad_left, pad_top, window_mask_indices[wi])
            edge_info_per_window.append(edge_info)
        
        def get_slot_feature(wi, c):
            start_idx, end_idx = window_mask_indices[wi]
            rel_idx = c - start_idx - 1
            return slot_features[wi, rel_idx, :]  

        merged_mask_channels = set()

        for (wi, wj, direction) in adjacency:
            start_i, end_i = window_mask_indices[wi]
            start_j, end_j = window_mask_indices[wj]
            
            if direction == 'horizontal':
                masks_wi = edge_info_per_window[wi]['right']
                masks_wj = edge_info_per_window[wj]['left']
            else:
                masks_wi = edge_info_per_window[wi]['bottom']
                masks_wj = edge_info_per_window[wj]['top']
            
            masks_wi = [m for m in masks_wi if m not in merged_mask_channels]
            masks_wj = [m for m in masks_wj if m not in merged_mask_channels]

            for ci in masks_wi:
                if ci in merged_mask_channels or ci % masks_per_window == 0:
                    continue
                for cj in masks_wj:
                    if cj in merged_mask_channels or cj % masks_per_window == 0:
                        continue
                    fi = get_slot_feature(wi, ci)
                    fj = get_slot_feature(wj, cj)
                    sim = self._cosine_similarity(fi, fj)
                    if sim > self.similarity:
                        keep = min(ci, cj)
                        remove = max(ci, cj)
                        
                        masks[:, keep] = masks[:, keep] + masks[:, remove]
                        masks[:, remove] = 0
                        
                        merged_mask_channels.add(remove)
        
        return masks

    def forward(self, masks: torch.Tensor, slot_features: torch.Tensor, keys) -> torch.Tensor:
        if slot_features.shape[0] != self.n_windows:
            assert slot_features.shape[0] % self.n_windows == 0
            new_shape = (self.n_windows, slot_features.shape[0] // self.n_windows) + slot_features.shape[1:]
            slot_features = slot_features.view(new_shape)        
        assert len(slot_features) == self.n_windows
        
        masks = self.to_one_hot(masks)
        pad_left, pad_top = self._parse_keys(keys)
        
        n_masks = masks.shape[1]
        window_mask_indices = self._get_window_indices(n_masks)
        
        adjacency = self._find_adjacent_windows(pad_left, pad_top)
        
        masks = self._merge_across_boundaries(masks, slot_features, pad_left, pad_top, adjacency, window_mask_indices)

        return masks
