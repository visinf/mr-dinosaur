"""Utility functions used in metrics computation."""
import torch
from typing import Optional
import numpy as np

def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)

def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor,combination_num: Optional[int] = None,fg=None,num=0) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).
    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)
    if combination_num is not None:
        background_threshold = n_pred_clusters // combination_num
        is_background = (pred_cluster_ids % background_threshold == 0)
        pred_cluster_ids[is_background] = 0
    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)
    channel_sums = pred_mask_oh.sum(dim=(0, 1)) 

    non_zero_channels = channel_sums > 0

    pred_mask_oh = pred_mask_oh[:, :, non_zero_channels]
    n_pred_clusters=pred_mask_oh.shape[-1]
   
    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)
    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex+0.000000000001
    ari = (rindex - expected_rindex) / denominator
    return torch.where(denominator > 0, ari, torch.ones_like(ari))



def fg_adjusted_rand_index(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0, combination_num: Optional[int] = None,num=0  # New parameter with default value
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probxbilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters)
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1:]), dim=-1
        )
    
    return adjusted_rand_index(pred_mask, true_mask_only_fg,combination_num,fg=True,num=num)



def _all_equal_masked(values: torch.Tensor, mask: torch.Tensor, dim=-1) -> torch.Tensor:
    """Check if all masked values along a dimension of a tensor are the same.

    All non-masked values are considered as true, i.e. if no value is masked, true is returned
    for this dimension.
    """
    assert mask.dtype == torch.bool
    _, first_non_masked_idx = torch.max(mask, dim=dim)

    comparison_value = values.gather(index=first_non_masked_idx.unsqueeze(dim), dim=dim)

    return torch.logical_or(~mask, values == comparison_value).all(dim=dim)



def get_pred_gt_scores(pred_mask: torch.Tensor, true_mask: torch.Tensor,width:int,combination_num: Optional[int] = None,iou_threshold=0.5,num=0):
    """
    Compute the Average Precision (AP) at a given IoU threshold.
    
    Args:
    - pred_mask_one_hot (np.ndarray): Prediction masks (batch_size, n_points, n_pred_clusters).
    - truth_mask (np.ndarray): Ground truth masks (batch_size, n_points, n_truth_clusters).
    - iou_threshold (float): IoU threshold to consider a prediction as a true positive.
    
    Returns:
    - average_precision (float): The average precision at the specified IoU threshold.
    """

    assert true_mask.shape[1]%width==0

    H=int(true_mask.shape[1]/width)

    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)
    if combination_num is not None:
        background_threshold = n_pred_clusters // combination_num
        is_background = (pred_cluster_ids % background_threshold == 0)
        pred_cluster_ids[is_background] = 0
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    non_zero_channels = (pred_mask_oh.sum(dim=(0, 1)) > 0)  # Check if the sum along the batch and height/width is non-zero
    pred_mask_oh = pred_mask_oh[..., non_zero_channels]  # Keep only the non-zero channels

    K=pred_mask_oh.shape[-1]
    C=true_mask_oh.shape[-1]
    pred_mask_oh=pred_mask_oh.squeeze(0).reshape(H,width,K)
    true_mask_oh=true_mask_oh.squeeze(0).reshape(H,width,C)
    true_mask_oh=true_mask_oh.cpu().to(torch.float16)
    pred_mask_oh=pred_mask_oh.cpu().to(torch.float16)
    
    gt=[true_mask_oh[:,:,s] for s in range(1,true_mask_oh.shape[2]) if true_mask_oh[:,:,s].max().item()>0]
    # remove the background slot
    if combination_num is not None:
        active_slots= [pred_mask_oh[:,:,s]for s in range(1,pred_mask_oh.shape[-1])]
        score=[1 for _ in range(1,pred_mask_oh.shape[-1])]
    else:
        active_slots= [pred_mask_oh[:,:,s]for s in range(pred_mask_oh.shape[-1])]
        score=[1 for s in range(pred_mask_oh.shape[-1])]
    del true_mask_oh,pred_mask_oh
    return gt,active_slots,score



def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def calculate_iou(mask1, set_masks):
    # Initialize a list to store the IoU values
    iou_scores = []

    # Convert masks to NumPy arrays
    mask1 = np.array(mask1, dtype=bool)

    for mask2 in set_masks:
        mask2 = np.array(mask2, dtype=bool)

        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)

        # Compute IoU
        iou = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou)

    return iou_scores

def voc_eval(gt_masks,
               pred_scores, pred_masks,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    
    nb_images = len(pred_masks) #added
    image_ids = []
    class_recs = {}
    nb_gt = 0

    for im in range(nb_images):
        image_ids += [im]*len(pred_masks[im])
        class_recs[im] = [False] * len(gt_masks[im]) 
        nb_gt += len(gt_masks[im])
    
    
    
    # flatten preds and scores
    pred_scores_flat = np.array([item for sublist in pred_scores for item in sublist])
    

    pred_masks_flat = np.stack([item for sublist in pred_masks for item in sublist])
    
    # sort by confidence
    sorted_ind = np.argsort(-pred_scores_flat)
    pred_masks_flat = pred_masks_flat[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        mask = pred_masks_flat[d]
        ovmax = -np.inf
        MASKSGT = gt_masks[image_ids[d]]

        
        # compute overlaps
        overlaps = calculate_iou(mask, MASKSGT)

        # case of no gt in one frame
        if not len(overlaps):
            ovmax = 0
        else:
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        
        if ovmax > ovthresh:
                if not R[jmax]:
                    tp[d] = 1.
                    R[jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(nb_gt)
    
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
