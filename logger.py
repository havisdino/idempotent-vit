import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log(self, **kwargs):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
class TensorBoardLogger(Logger):
    def __init__(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.writer = SummaryWriter('logs')
    
    def log(self, **kwargs):
        global_step = kwargs.get('global_step')
        del kwargs['global_step']
        
        for tag, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                self.writer.add_images(tag, value, global_step)
            else:
                self.writer.add_scalar(tag, value, global_step)
            
    def close(self):
        self.writer.close()