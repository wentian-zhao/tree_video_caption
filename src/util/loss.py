import torch
from torch.autograd import Variable
import torch.nn.functional as F


def sequence_mask(seq_length, max_length=None, invert=False, dtype=torch.bool):
    """

    :param seq_length: (batch_size,)
    :param max_length: int
    :return:
    """
    batch_size = seq_length.shape[0]
    if max_length is None:
        max_length = torch.max(seq_length)
    mask = torch.arange(start=0, end=max_length, dtype=seq_length.dtype, device=seq_length.device)
    mask = mask.unsqueeze(0).expand(batch_size, max_length)
    mask = (mask < seq_length.unsqueeze(1))
    if invert:
        mask = torch.logical_not(mask)
    mask = mask.to(dtype)
    return mask


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: FloatTensor, (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: LongTensor, (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    device = logits.device
    batch_size, max_len, vocab_size = logits.shape
    assert target.dim() == 2 and batch_size == target.shape[0] and max_len == target.shape[1]
    assert length.dim() == 1 and batch_size == length.shape[0]

    logits_flat = logits.reshape(batch_size * max_len, vocab_size)

    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    target_flat = target.reshape(batch_size * max_len, 1)

    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)   # (batch_size * max_len, 1)

    losses = losses_flat.reshape(batch_size, max_len)   # (batch_size, max_len)

    mask = sequence_mask(seq_length=length, max_length=max_len)   # (batch_size, max_len)
    mask = mask.to(losses.device)
    losses = losses * mask          # (batch_size, max_len)
    loss = losses.sum() / length.float().sum().to(device)
    return loss


def masked_bce_with_logits(logits, target, length, weight_pos=1., weight_neg=1.):
    """

    :param logits: (batch_size, max_len)
    :param target: (batch_size, max_len)
    :param length: (batch_size,)
    :return:
    """
    device = logits.device
    batch_size, max_len = logits.shape
    assert target.dim() == 2 and batch_size == target.shape[0] and max_len == target.shape[1]
    assert length.dim() == 1 and batch_size == length.shape[0]

    mask = sequence_mask(seq_length=length, max_length=max_len)  # (batch_size, max_len)
    mask = mask.to(device)

    _target = target.to(torch.bool)
    pos_mask = (mask * _target).to(torch.float)
    neg_mask = (mask * torch.logical_not(_target)).to(torch.float)

    _mask = pos_mask * weight_pos + neg_mask * weight_neg

    loss = F.binary_cross_entropy_with_logits(logits, target.to(torch.float), weight=_mask, reduction='mean')

    return loss

