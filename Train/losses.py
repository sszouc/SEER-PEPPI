# Train/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AUCMinMaxLoss(nn.Module):
    """
    AUC Min-Max Margin Loss for link prediction.

    From公式(4):
    L_AUC(w) = E[(g_w(x) - a(w))^2 | y=1]
             + E[(g_w(x') - b(w))^2 | y'=1]
             + max_{α≥0} 2α(m - a(w) + b(w)) - α^2

    where:
        a(w) = E[g_w(x) | y=1]  (positive samples mean)
        b(w) = E[g_w(x') | y'=1] (negative samples mean)
        m: margin hyperparameter
    """

    def __init__(self, margin=1, alpha=None):
        """
        Args:
            margin (float): Expected margin between a(w) and b(w), defaults to 1
            alpha (float, optional): Lagrangian multiplier. If None, compute optimal alpha
        """
        super(AUCMinMaxLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # Will be optimized if None

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores (torch.Tensor): Predictions for positive samples (shape: [n_pos])
            neg_scores (torch.Tensor): Predictions for negative samples (shape: [n_neg])

        Returns:
            torch.Tensor: AUC loss value
        """
        # Ensure inputs are 1D tensors
        pos_scores = pos_scores.view(-1)
        neg_scores = neg_scores.view(-1)

        # Compute a(w) and b(w) - empirical means
        a = pos_scores.mean()  # E[g_w(x) | y=1]
        b = neg_scores.mean()  # E[g_w(x') | y'=1]

        # Compute variance terms: E[(g_w(x) - a)^2] and E[(g_w(x') - b)^2]
        pos_var = ((pos_scores - a) ** 2).mean()
        neg_var = ((neg_scores - b) ** 2).mean()

        # Compute the margin violation term
        margin_violation = self.margin - a + b

        if margin_violation <= 0:
            # Margin satisfied, only the variance terms matter
            alpha_opt = 0.0
            hinge_term = 0.0
        else:
            # Compute optimal alpha (closed-form solution for the max problem)
            # The max over α≥0 of 2α*(margin_violation) - α^2
            # Derivative: 2*margin_violation - 2α = 0 => α_opt = margin_violation
            alpha_opt = margin_violation
            hinge_term = 2 * alpha_opt * margin_violation - alpha_opt ** 2

        # Store alpha for debugging
        self.alpha_opt = alpha_opt

        # Total AUC loss (公式4)
        loss = pos_var + neg_var + hinge_term

        return loss


class DualViewLoss(nn.Module):
    """
    Combined BCE + AUC loss for link prediction.

    From公式(5): L = η * L_BCE + (1-η) * L_AUC

    This class efficiently combines both losses, especially for batch training.
    """

    def __init__(self, eta=0.3, auc_margin=1, pos_weight=None):
        """
        Args:
            eta (float): Weight for BCE loss (between 0 and 1).
                        L = eta * L_BCE + (1-eta) * L_AUC
            auc_margin (float): Margin parameter m for AUC loss
            pos_weight (torch.Tensor, optional): Positive weight for BCE (for imbalanced data)
        """
        super(DualViewLoss, self).__init__()
        assert 0 <= eta <= 1, "eta must be between 0 and 1"

        self.eta = eta
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.auc_loss = AUCMinMaxLoss(margin=auc_margin)

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor): Raw logits from model (before sigmoid)
            labels (torch.Tensor): Binary labels (0 or 1)

        Returns:
            tuple: (total_loss, bce_loss, auc_loss)
        """
        # Split into positive and negative samples for AUC loss
        pos_mask = labels == 1
        neg_mask = labels == 0

        # Handle edge cases
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            # Fallback to BCE if no positive or negative samples in batch
            auc_loss_value = torch.tensor(0.0, device=logits.device)
        else:
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]
            auc_loss_value = self.auc_loss(pos_logits, neg_logits)

        # BCE loss
        bce_loss_value = self.bce_loss(logits, labels.float())

        # Combined loss
        if self.eta == 1.0:
            total_loss = bce_loss_value
        elif self.eta == 0.0:
            total_loss = auc_loss_value
        else:
            total_loss = self.eta * bce_loss_value + (1 - self.eta) * auc_loss_value

        return total_loss, bce_loss_value, auc_loss_value


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with optional symmetric direction.
    """

    def __init__(self, temperature=0.2, symmetric=True, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.eps = eps

    def forward(self, z1, z2):
        if z1.shape != z2.shape:
            raise ValueError("z1 and z2 must have the same shape")
        z1 = F.normalize(z1, dim=1, eps=self.eps)
        z2 = F.normalize(z2, dim=1, eps=self.eps)
        logits = torch.matmul(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_1 = F.cross_entropy(logits, labels)
        if not self.symmetric:
            return loss_1
        loss_2 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_1 + loss_2)


# class EffectiveDualViewLoss(DualViewLoss):
#     """
#     Enhanced version with adaptive margin and stability improvements.
#     """
#
#     def __init__(self, eta=0.5, auc_margin=0.5, pos_weight=None,
#                  use_squared_hinge=False, adaptive_margin=False):
#         """
#         Args:
#             eta: Weight for BCE loss
#             auc_margin: Base margin parameter m
#             pos_weight: Positive weight for BCE
#             use_squared_hinge: Use squared hinge loss for AUC (more stable)
#             adaptive_margin: Adapt margin based on current performance
#         """
#         super(EffectiveDualViewLoss, self).__init__(eta, auc_margin, pos_weight)
#         self.use_squared_hinge = use_squared_hinge
#         self.adaptive_margin = adaptive_margin
#         self.moving_a = None  # For adaptive margin
#         self.moving_b = None
#         self.momentum = 0.99
#
#     def _compute_adaptive_margin(self, a, b):
#         """Adapt margin based on current gap"""
#         if self.moving_a is None:
#             self.moving_a = a.detach()
#             self.moving_b = b.detach()
#         else:
#             self.moving_a = self.momentum * self.moving_a + (1 - self.momentum) * a.detach()
#             self.moving_b = self.momentum * self.moving_b + (1 - self.momentum) * b.detach()
#
#         # Increase margin if the gap is already large
#         current_gap = self.moving_a - self.moving_b
#         if current_gap > self.margin:
#             return current_gap * 0.5  # Target half of current gap
#         return self.margin
#
#     def forward(self, logits, labels):
#         pos_mask = labels == 1
#         neg_mask = labels == 0
#
#         if pos_mask.sum() == 0 or neg_mask.sum() == 0:
#             bce_loss_value = self.bce_loss(logits, labels.float())
#             return bce_loss_value, bce_loss_value, torch.tensor(0.0, device=logits.device)
#
#         pos_logits = logits[pos_mask]
#         neg_logits = logits[neg_mask]
#
#         # Adaptive margin
#         if self.adaptive_margin:
#             current_a = pos_logits.mean()
#             current_b = neg_logits.mean()
#             current_margin = self._compute_adaptive_margin(current_a, current_b)
#             self.auc_loss.margin = current_margin
#
#         # Compute AUC loss
#         a = pos_logits.mean()
#         b = neg_logits.mean()
#         pos_var = ((pos_logits - a) ** 2).mean()
#         neg_var = ((neg_logits - b) ** 2).mean()
#
#         margin_violation = self.auc_loss.margin - a + b
#
#         if margin_violation <= 0:
#             hinge_term = 0.0
#         else:
#             alpha_opt = margin_violation
#             if self.use_squared_hinge:
#                 # Squared hinge for smoother gradient
#                 hinge_term = (margin_violation ** 2)
#             else:
#                 hinge_term = 2 * alpha_opt * margin_violation - alpha_opt ** 2
#
#         auc_loss_value = pos_var + neg_var + hinge_term
#
#         # BCE loss
#         bce_loss_value = self.bce_loss(logits, labels.float())
#
#         # Combined
#         total_loss = self.eta * bce_loss_value + (1 - self.eta) * auc_loss_value
#
#         return total_loss, bce_loss_value, auc_loss_value