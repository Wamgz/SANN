# https://github.com/alfonmedela/triplet-loss-pytorch
import torch
from torch import nn
from torch.nn import functional as F
def cross_entropy_loss(pred_class_logits, gt_classes, eps=0.3, alpha=0.2, reduction='none'):
    num_classes = pred_class_logits.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_logits, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = F.log_softmax(pred_class_logits, dim=1)
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    if reduction is not None:
        loss = loss.sum() / non_zero_cnt
    return loss

class ContrastLoss(nn.Module):
    
    def __init__(self):
        super(ContrastLoss, self).__init__()
        pass

    def forward(self, anchor_fea, reassembly_fea, contrast_label):
        contrast_label = contrast_label.float()
        anchor_fea = anchor_fea.detach()
        loss = -(F.cosine_similarity(anchor_fea, reassembly_fea, dim=-1))
        loss = loss*contrast_label
        return loss.mean()
class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, cue, label):
        cue = label.reshape(-1, 1, 1, 1) * cue 
        num_reg = (
            torch.sum(label) * cue.shape[1] * cue.shape[2] * cue.shape[3]
        ).type(torch.float) # tensor(196608., device='cuda:0')
        reg_loss = (
            torch.sum(torch.abs(cue)) / (num_reg + 1e-9)
        )
        return reg_loss


def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(
        pairwise_distances_squared, torch.tensor([0.0]).to(device)
    )
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.0
    error_mask[error_mask <= 0.0] = 0.0

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones(
        (pairwise_distances.shape[0], pairwise_distances.shape[1])
    ) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(
        pairwise_distances.to(device), mask_offdiagonals.to(device)
    )
    return pairwise_distances


def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = (
        torch.min(
            torch.mul(pdist_matrix_tile - axis_maximums[0], mask),
            dim=1,
            keepdim=True,
        )[0]
        + axis_maximums[0]
    )
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = (
        torch.max(
            torch.mul(pdist_matrix - axis_minimums[0], adjacency_not),
            dim=1,
            keepdim=True,
        )[0]
        + axis_minimums[0]
    )
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(
        mask_final, negatives_outside, negatives_inside
    )

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(
        torch.ones(batch_size)
    ).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (
        torch.max(
            torch.mul(loss_mat, mask_positives), torch.tensor([0.0]).to(device)
        )
    ).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, x, target, **kwargs):
        x = F.adaptive_avg_pool2d(x, [1, 1]).view(x.shape[0], -1)
        return TripletSemiHardLoss(
            target, F.normalize(x, p=2, dim=-1), target.device, self.margin
        )