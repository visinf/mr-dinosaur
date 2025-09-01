"""Metrics related to the evaluation of masks."""
from typing import Optional

import torch
import torchmetrics

from ocl.metrics.utils import adjusted_rand_index, fg_adjusted_rand_index, tensor_to_one_hot,get_pred_gt_scores, voc_eval, compute_f1
from ocl.utils.resizing import resize_patches_to_image

class APandARMetric(torchmetrics.Metric):
    """Computes Average Precision for masks."""

    def __init__(
        self,
        convert_target_one_hot: bool = False,
        ignore_overlaps: bool = False,
        combination_num: Optional[int] = None,
        num:int=0,
        ap: bool=False,
        ar: bool=False
        ):
        super().__init__()
        self.combination_num=combination_num
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.num=num
        self.ap=ap
        self.ar=ar
        self.gts=[]
        self.preds=[]
        self.scores=[]

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
       
        width=target.shape[-1]
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0
        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"
        gt,active_slots,score= get_pred_gt_scores(prediction, target,width=width, combination_num=self.combination_num)
        self.gts.append(gt)
        self.preds.append(active_slots)
        self.scores.append(score)
        del gt, active_slots, score

    def compute(self):
        rec, prec = voc_eval(self.gts, self.scores, self.preds, 0.5)
        f1_score = compute_f1(rec[-1], prec[-1])
        if self.ap:
            return torch.tensor(prec[-1])
        elif self.ar:
            return torch.tensor(rec[-1])
        else:
            return torch.tensor(f1_score)
        
class ARIMetric(torchmetrics.Metric):
    """Computes ARI metric."""

    def __init__(
        self,
        foreground: bool = True,
        convert_target_one_hot: bool = False,
        ignore_overlaps: bool = False,
        combination_num: Optional[int] = None,
        num: int=0,
        ):
        super().__init__()
        self.foreground = foreground
        self.combination_num=combination_num
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num=num
    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0
        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)
        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh
        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"
        if target.shape[-1] > 2:
            if self.foreground:
                ari = fg_adjusted_rand_index(prediction, target, combination_num=self.combination_num,num=self.num)
            else:
                ari = adjusted_rand_index(prediction, target, combination_num=self.combination_num,num=self.num)
            self.values += ari.sum()
            self.total += len(ari)
        else:
            self.values += 0
            self.total += 0
        self.num+=1

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class PatchARIMetric(ARIMetric):

    def __init__(
        self,
        foreground=True,
        resize_masks_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(foreground=foreground, **kwargs)
        self.resize_masks_mode = resize_masks_mode

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, P) or (B, F, C, P), where C is the
                number of classes and P the number of patches.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        h, w = target.shape[-2:]
        assert h == w

        prediction_resized = resize_patches_to_image(
            prediction, size=h, resize_mode=self.resize_masks_mode
        )

        return super().update(prediction=prediction_resized, target=target)