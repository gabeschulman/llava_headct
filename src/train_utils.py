from torch import Tensor


def determine_is_resume(config_name: str) -> bool:
    return True if "pretrain" in config_name else False


def get_training_weight_config(
    epoch: int, global_batch_idx: int, sample_weights: Tensor, is_resume: bool = False
) -> tuple[Tensor, float]:
    """
    Get training hyperparameters based on epoch and whether resuming.

    Args:
        epoch: Current epoch number (1-indexed)
        global_batch_idx: Batch count within current epoch
        is_resume: True if loading from checkpoint (Epoch 3+)
    """

    if is_resume or epoch > 1:
        sample_weights_adjusted = sample_weights
    else:
        warmup_batches = 8000
        if global_batch_idx < warmup_batches:
            scale_factor = global_batch_idx / warmup_batches
            sample_weights_adjusted = 1.0 + (sample_weights - 1.0) * scale_factor
        else:
            sample_weights_adjusted = sample_weights

    if is_resume or epoch > 1:
        contrastive_weight = 0.3
    else:
        if global_batch_idx < 5000:
            contrastive_weight = 2.0
        elif global_batch_idx < 8000:
            contrastive_weight = 1.0
        else:
            contrastive_weight = 0.5

    return sample_weights_adjusted, contrastive_weight
