from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import torch
from sklearn.metrics import roc_auc_score, f1_score

from .tensor_utils import center_crop_reshape


def compute_batched_precision_recall_f1(occ_maps, gt_masks):
    """
    Compute precision, recall, and F1 score over a batch of binary masks.
    
    Args:
        occ_maps: torch.Tensor [B,H,W], predicted binary masks (0 or 1)
        gt_masks: torch.Tensor [B,H,W], ground truth binary masks (0 or 1)
        
    Returns:
        precision, recall, f1: floats, averaged over batch
    """
    B = occ_maps.size(0)
    precisions, recalls, f1s = [], [], []
    
    for i in range(B):
        pred = occ_maps[i].cpu().numpy().flatten()
        gt = gt_masks[i].cpu().numpy().flatten()
        
        precisions.append(precision_score(gt, pred, zero_division=0))
        recalls.append(recall_score(gt, pred, zero_division=0))
        f1s.append(f1_score(gt, pred, zero_division=0))
    
    return sum(precisions) / B, sum(recalls) / B, sum(f1s) / B


def compute_batched_auroc(anomaly_scores, gt_masks):
    """
    Compute AUROC (Area Under ROC Curve) for anomaly scores.
    
    Args:
        anomaly_scores: torch.Tensor [B,H,W], continuous anomaly scores (float)
        gt_masks: torch.Tensor [B,H,W], binary ground truth masks (0 or 1)
        
    Returns:
        auroc: float, averaged over batch
    """
    B = anomaly_scores.size(0)
    aurocs = []
    
    for i in range(B):
        scores = anomaly_scores[i].cpu().numpy().flatten()
        gt = gt_masks[i].cpu().numpy().flatten()
        
        try:
            auroc = roc_auc_score(gt, scores)
        except ValueError:
            # In case gt contains only one class (no positives or no negatives)
            auroc = float('nan')
        aurocs.append(auroc)
    
    # Filter out nan values before averaging
    aurocs = [a for a in aurocs if not (a != a)]  # remove nan
    
    if len(aurocs) == 0:
        return float('nan')
    
    return sum(aurocs) / len(aurocs)


def compute_topk_accuracy(predictions, k_values=[1, 3, 5]):
    """
    Calculate top-k accuracy for multiple k values.

    Args:
        predictions: List of prediction batches
        k_values: List of k values to calculate accuracy for

    Returns:
        Dict mapping k -> accuracy
    """
    total_samples = 0
    correct_predictions = {k: 0 for k in k_values}

    for batch in predictions:
        batch_size = len(batch["path"])
        total_samples += batch_size

        for i in range(batch_size):
            gt_label = batch["gt_label"][i]
            gt_label_idx = gt_label
            pred_logits = batch["pred_probs"][i]#.cpu()

            # Get indices of top-k predictions
            topk_values, topk_indices = torch.topk(pred_logits, max(k_values))

            # Check if ground truth is in top-k predictions for each k
            for k in k_values:
                if gt_label_idx in topk_indices[:k]:
                    correct_predictions[k] += 1

    # Calculate accuracy for each k
    accuracy = {
        k: correct_predictions[k] / total_samples * 100 if total_samples > 0 else 0
        for k in k_values
    }

    return accuracy


def eval_segmentation(ground_truth, anomaly_map, occ_map):
    if ground_truth.shape[1] != 224 | ground_truth.shape[2] != 224:
        ground_truth = center_crop_reshape(ground_truth, (224,224))

    assert (ground_truth.shape == occ_map.shape == anomaly_map.shape)

    # Flatten
    ground_truth = ground_truth.flatten()
    occ_map = occ_map.flatten()
    anomaly_map = anomaly_map.flatten()

    tp = ((occ_map == 1) & (ground_truth == 1)).sum().item()
    tn = ((occ_map == 0) & (ground_truth == 0)).sum().item()
    fp = ((occ_map == 1) & (ground_truth == 0)).sum().item()
    fn = ((occ_map == 0) & (ground_truth == 1)).sum().item()

    precision = tp / (tp + fp + 1e-9)
    recall  = tp / (tp + fn + 1e-9)

    iou  = tp / (tp + fp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    pos_frac = occ_map.float().mean()  # predicted area %

    # Compute pixel-level AUROC
    auroc_px = roc_auc_score(ground_truth, anomaly_map)

    if (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return dict(
        precision=precision, 
        recall=recall, 
        f1=f1_score, 
        auroc=auroc_px, 
        iou=iou, 
        fpr=fpr, 
        pred_area=pos_frac
    )
