import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_loss(pred, target):
    """
    Compute the scale-invariant depth loss between MiDaS-style prediction and GT.
    Scale alignment (median scaling) is applied before computing log loss.
    """

    print(f"shape{pred.shape}")
    print(f"shape{target.shape}")


    valid = torch.logical_and(target > 1, target < 80)

    pred_valid = pred[valid]
    target_valid = target[valid]

    #print(f"?? min max of pred: {pred.min()} {pred.max()}")
    #print(f"?? min max of target: {target.min()} {target.max()}")

    #pred_valid = pred
    #target_valid = target

    #pred_valid = pred
    #target_valid = target

    if pred_valid.numel() == 0:
        return 0.0 * pred.sum()

    eps = 1e-6
    pred_valid = torch.clamp(pred_valid, min=eps)
    target_valid = torch.clamp(target_valid, min=eps)

    #g   = torch.log(pred_valid) - torch.log(target_valid)   # Δlog
    #Dg  = torch.var(g) + 0.15 * torch.mean(g).pow(2)
    #loss = 10 * torch.sqrt(Dg + 1e-4)           # ε 로 0‑division 방지

    # ✅ MiDaS 스타일 scale alignment
    #scale = torch.median(target_valid) / torch.median(pred_valid)
    #pred_valid = pred_valid * scale

    # ✅ log difference loss
    g = torch.log(pred_valid) - torch.log(target_valid)
    Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
    loss = 10 * torch.sqrt(Dg)
    
    return loss
# def depth_loss(pred, target):
#     """
#     Compute the depth loss between the predicted and target depth maps.
#     :param pred: Predicted depth map
#     :param target: Target depth map
#     :return: Depth loss
#     """

#     #diff = torch.abs(pred - target)
#     #loss = torch.mean(diff)

#     #valid = target > 0
#     #valid = np.logical_and(40 > target, target > 0)
#     valid = torch.logical_and(target > 0, target < 40)

#     pred_valid = pred[valid]
#     target_valid = target[valid]
    
#     g = torch.log(pred_valid) - torch.log(target_valid)
#     # n, c, h, w = g.shape
#     # norm = 1/(h*w)
#     # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

#     Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
#     loss = 10 * torch.sqrt(Dg)
#     return loss

def l2_loss(pred, target):
    """
    Compute masked L2 loss (MSE) between predicted and target depth maps.
    Only valid pixels within range (0, 40) are considered.

    Args:
        pred (Tensor): Predicted depth map (B, 1, H, W)
        target (Tensor): Ground truth depth map (B, 1, H, W)

    Returns:
        Tensor: Scalar loss
    """


    # L2 loss (mean squared error)
    loss = torch.mean((pred_valid - target_valid) ** 2)

    return loss





def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss
