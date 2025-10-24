import numpy as np

def wd_cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_wd_scheduler(config, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    base_value = config.TRAIN.WEIGHT_DECAY
    final_value = config.TRAIN.WEIGHT_DECAY_END
    epochs = config.TRAIN.MAX_EPOCHS
    
    return wd_cosine_scheduler(base_value, final_value, epochs, \
        niter_per_ep, warmup_epochs, start_warmup_value)