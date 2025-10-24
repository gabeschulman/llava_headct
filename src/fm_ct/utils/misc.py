import os
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import torch
import torch.nn as nn
import torch.distributed as dist

from src.fm_ct.utils.pos_embed import interpolate_pos_embed

def create_dataset(images, labels):
    dataset = []
    
    if labels is None:
        for img in images:
            sample_dict = dict()
            sample_dict['image'] = img
            dataset.append(sample_dict)
    else:
        for img, label in zip(images, labels):
            sample_dict = dict()
            sample_dict['image'] = img
            sample_dict['pred_label'] = label
            dataset.append(sample_dict)
            
    return dataset


def save_checkpoint(model, momentum_model, epoch, optimizer, scheduler, \
    filename="model.pt", best_loss=0, dir_add=None, logger=None):
    
    model_state_dict = model.state_dict()
    
    if momentum_model != None:
        momentum_model_state_dict = momentum_model.state_dict()
    else:
        momentum_model_state_dict = None
        
    optimizer_dict = optimizer.state_dict()
    scheduler_dict = scheduler.state_dict()
    save_dict = {"epoch": epoch, "best_loss": best_loss, "state_dict": model_state_dict, \
        "momentum_model_state_dict": momentum_model_state_dict, "optimizer": optimizer_dict, \
        "scheduler": scheduler_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    logger.info(f"Saving checkpoint {filename}")
    
    
def load_optimizer(optimizer, scheduler, loaded_state_dict, logger=None):
    epoch = 0
    if 'optimizer' in loaded_state_dict.keys():
        msg = optimizer.load_state_dict(loaded_state_dict['optimizer'])
        logger.info(f"Loaded optimizer state: {msg}")
    
    if 'scheduler' in loaded_state_dict.keys():
        msg = scheduler.load_state_dict(loaded_state_dict['scheduler'])
        logger.info(f"Loaded scheduler state: {msg}")
        
    if 'epoch' in loaded_state_dict.keys():
        epoch = loaded_state_dict['epoch']
        logger.info(f"Loaded epoch: {epoch}")
            
    return optimizer, scheduler, epoch


def load_model(config, model, momentum_model, logger, model_name="dino"):
    torch.serialization.add_safe_globals([(np.core.multiarray, 'scalar')])
    # Load model with wrong size weights unloaded
    if config.MODEL.PRETRAINED != None:
        all_state_dicts = torch.load(config.MODEL.PRETRAINED, \
            map_location=torch.device('cpu'))
        # model load
        loaded_state_dict = all_state_dicts['state_dict']
        new_state_dict = {k.replace("module.", "").replace("backbone.", "").replace("_orig_mod.", ""): \
            v for k, v in loaded_state_dict.items()}
        # # interpolate position embedding
        # interpolate_pos_embed(model, new_state_dict)
        msg = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Load Pretrained Model: {msg} for Achitecture: {config.MODEL.NAME}")
        # momentum model load
        if momentum_model != None:
            momentum_loaded_state_dict = all_state_dicts['momentum_model_state_dict']
            momentum_new_state_dict = {k.replace("module.", "").replace("backbone.", "").replace("_orig_mod.", ""): \
                v for k, v in momentum_loaded_state_dict.items()}
            msg = momentum_model.load_state_dict(momentum_new_state_dict, strict=False)
            logger.info(f"Load Pretrained Momentum Model: {msg} for Achitecture: {config.MODEL.NAME}")

        return all_state_dicts
    else:
        return None


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
                
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
        

class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def all_reduce_mean(x):
    if not is_dist_avail_and_initialized():
        world_size =  1
    else:
        world_size = dist.get_world_size()
        
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    
    
def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def cleanup():
    dist.destroy_process_group()


def all_gather(tensor):
    return AllGatherFunction.apply(tensor)


def set_requires_grad_false(*models, lora=False):
    if lora:
        for model in models:
            for name, param in model.named_parameters():
                is_trainable = (
                    "lora" in name or
                    "bias" in name or
                    "embeddings" in name or
                    "norm" in name
                )
                param.requires_grad = is_trainable
    else:
        for model in models:
            for param in model.parameters():
                param.requires_grad = False
            
            
def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None
            
            
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

            
@torch.no_grad()
def _update_momentum_encoder(model: torch.nn.Module, momentum_model: torch.nn.Module, m: float) -> None:
    """
    Momentum update of the momentum encoder.

    Args:
        model (torch.nn.Module): The main model.
        momentum_model (torch.nn.Module): The momentum model.
        m (float): Momentum coefficient.
    """
    for param_q, param_k in zip(model.parameters(), momentum_model.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)
    

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        cls_feature = output[:, 0, :]
        
        return {'dino_output': self.head(cls_feature)}
    

def plot_regression(x, y, title, percent="None"):
    # Create a scatter plot
    plt.figure(figsize=(20, 15))
    plt.scatter(x, y, label='data points', marker='o')

    # Determine the range to plot the diagonal
    # Use the combined range of x_data and y_data for the diagonal
    min_val = min(x)
    max_val = max(x)

    # Plot the diagonal line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')

    # Optionally, set the axis limits if you want to enforce equal aspect ratio
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Add titles and labels
    plt.title(f'Plot of {title}')
    plt.xlabel('Target')
    plt.ylabel('Prediction')

    # Show legend
    plt.legend()
    
    # Save plot
    plt.savefig(f'regression_plot_{percent}.png', dpi=300)
    

def plot_pr_curve(targets, preds, percent="None"):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    # Computing ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(targets, preds)
    # Calculate the area under the Precision-Recall curve
    average_precision = average_precision_score(targets, preds)
    plt.plot(recall, precision, label=f'AP={average_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="upper right")
    
    plt.savefig(f'../plots/roc_pr_curve_plot_{percent}.png', dpi=300)
