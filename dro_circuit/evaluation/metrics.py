"""Metric functions compatible with EAP-IG's expected signature.

EAP-IG metrics have the signature:
    metric(logits, clean_logits, input_lengths, labels) -> Tensor
"""

import torch
import torch.nn.functional as F


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_lengths: torch.Tensor,
    labels: torch.Tensor,
    loss: bool = True,
    mean: bool = True,
) -> torch.Tensor:
    """
    Logit difference: correct_logit - incorrect_logit at the final token.

    Args:
        logits: (batch, seq_len, vocab_size)
        clean_logits: (batch, seq_len, vocab_size) - reference from clean run
        input_lengths: (batch,) - length of each input
        labels: (batch, 2) with [correct_token_id, incorrect_token_id]
        loss: if True, negate (for minimization)
        mean: if True, return scalar mean
    """
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_logits = logits[batch_idx, input_lengths - 1]
    correct = final_logits[batch_idx, labels[:, 0]]
    incorrect = final_logits[batch_idx, labels[:, 1]]

    diff = correct - incorrect
    if loss:
        diff = -diff
    if mean:
        return diff.mean()
    return diff


def kl_divergence(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_lengths: torch.Tensor,
    labels: torch.Tensor,
    mean: bool = True,
) -> torch.Tensor:
    """
    KL(clean || circuit) at the final token position.

    Measures how much the circuit's output distribution diverges from the full model.
    """
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_circuit = logits[batch_idx, input_lengths - 1]
    final_clean = clean_logits[batch_idx, input_lengths - 1]

    kl = F.kl_div(
        F.log_softmax(final_circuit, dim=-1),
        F.softmax(final_clean, dim=-1),
        reduction="none",
    ).sum(dim=-1)

    if mean:
        return kl.mean()
    return kl


# Convenience aliases matching EAP-IG convention
def logit_diff_loss(logits, clean_logits, input_lengths, labels):
    """Logit diff as loss (negated, mean-reduced) for attribution scoring."""
    return logit_diff(logits, clean_logits, input_lengths, labels, loss=True, mean=True)


def logit_diff_metric(logits, clean_logits, input_lengths, labels):
    """Logit diff as metric (non-negated, mean-reduced) for evaluation."""
    return logit_diff(logits, clean_logits, input_lengths, labels, loss=False, mean=True)
