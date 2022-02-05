import time

import torch
from torch.nn.utils import clip_grad_norm, clip_grad_norm_
import torch.optim
from torch.optim import lr_scheduler


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def clip_gradient_norm(optimizer, max_grad_norm):
    for group in optimizer.param_groups:
        for param in group['params']:
            clip_grad_norm_(param, max_grad_norm)


def optimizer_add_args(optimizer_name, parser):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        parser.add_argument('--learning-rate', default=2e-4, type=float)
        parser.add_argument('--adam-betas', type=str, default="(0.9, 0.98)")
        parser.add_argument('--adam-eps', type=float, default=1e-8)
        parser.add_argument('--weight-decay', type=float, default=0)
    elif optimizer_name == 'sgd':
        parser.add_argument('--learning-rate', default=0.1, type=float)
        parser.add_argument('--weight-decay', type=float, default=1e-5)
        parser.add_argument('--momentum', type=float, default=0.95)


def get_optimizer(args, model):
    optimizer_name = args.optimizer.lower()
    if optimizer_name == 'adam':
        betas = eval(args.adam_betas)
        optimizer = torch.optim.Adam(params=model.parameters(),
                         lr=args.learning_rate, betas=betas, eps=args.adam_eps, weight_decay=args.weight_decay)
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    return optimizer


def scheduler_add_args(scheduler_name, parser):
    scheduler_name = scheduler_name.lower()

    parser.add_argument('--learning_rate_decay_start', default=-1, type=int)

    if scheduler_name == 'steplr':
        parser.add_argument('--learning_rate_decay_every', default=3, type=int)
        parser.add_argument('--learning_rate_decay_rate', default=0.8, type=float)
    elif scheduler_name == 'plateau':
        parser.add_argument('--reduce_factor', default=0.5, type=int)
        parser.add_argument('--patience_epoch', default=1, type=int)

def get_scheduler(args, optimizer):
    scheduler_name = args.scheduler.lower()
    if scheduler_name == 'steplr':
        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.learning_rate_decay_every,
            gamma=args.learning_rate_decay_rate
        )
    elif scheduler_name == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=args.reduce_factor,
            patience=args.patience_epoch,
            verbose=True
        )

    return scheduler

class Timer:
    def __init__(self):
        self.ticks = {}
        self.times = {}

    def clear(self):
        self.ticks.clear()
        self.times.clear()

    def tick(self, event=None):
        if event is None:
            event = '_default'
        self.ticks[event] = time.time()

    def tock(self, event, acc=True):
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        if event not in self.ticks:
            return
        if event not in self.times:
            self.times[event] = 0.
        if acc:
            self.times[event] += time.time() - self.ticks[event]
        else:
            self.times[event] = time.time() - self.ticks[event]
        del self.ticks[event]

    def get_time(self):
        return self.times


global_timer = Timer()