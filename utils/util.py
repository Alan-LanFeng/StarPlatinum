import os
import time
import pathlib
import inspect
import numpy as np
import settings as setin
from numba import jit
from collections import deque


def log(level, msg):
    func = inspect.currentframe().f_back.f_code
    fmt = "{date.tm_mon}{date.tm_mday} {date.tm_hour}:{date.tm_min}:{date.tm_sec} "\
          "{pid} {filename}:{lineno}] {msg}".format(
        date=time.localtime(time.time()), pid=os.getpid(),
        filename=pathlib.Path(func.co_filename).name,
        lineno=func.co_firstlineno, msg=msg)
    setin.LOGGER.log(level, fmt)

@jit(nopython=True)
def wrap_angle_zero_2pi(radians: np.ndarray):
    """wrap angle to [0, 2pi]"""
    return radians % (2 * np.pi)

@jit(nopython=True)
def wrap_angle_miuspi_pi(radians: np.ndarray):
    """wrap angle to [-pi, pi]"""
    return (radians + np.pi) % (2 * np.pi) - np.pi


def mkdir(dir_name: str):
    """make directory

    Args:
        dir_name (str): name of the damended directory
    """    
    if os.path.isdir(dir_name):
        log('info', '\033[93m{} has existed \033[0m'.format(dir_name))
    else:
        os.mkdir(dir_name)


def batchtocuda(batch):
    assert type(batch) == dict, "The batch data must be dictionary"
    for k, v in batch.items():
        if isinstance(v, list):
            for i, cv in enumerate(v):
                try:
                    batch[k][i] = cv.cuda()
                except AttributeError:
                    batch[k][i] = cv
        else:
            try:
                batch.update({k: v.cuda()})
            except AttributeError:
                batch.update({k: v})
    return batch


def cudatocpu(batch):
    assert type(batch) == dict, "The batch data must be dictionary"
    for k, v in batch.items():
        if isinstance(v, list):
            for i, cv in enumerate(v):
                try:
                    batch[k][i] = cv.cpu()
                except AttributeError:
                    batch[k][i] = cv
        else:
            try:
                batch.update({k: v.cpu()})
            except AttributeError:
                batch.update({k: v})
    return batch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    __slots__ = ('val', 'avg', 'sum', 'count', 'queue')
    
    def __init__(self, quelen=100):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.queue = deque(maxlen=quelen)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.queue.clear()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
        self.queue.appendleft(self.val)

    @property
    def qavg(self):
        return sum(self.queue) / len(self.queue)


