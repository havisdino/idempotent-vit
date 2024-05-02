from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from tqdm import tqdm

from sample import sample
from logger import Logger
from modules import ViT


@dataclass(eq=False)
class Trainer:
    f: ViT
    f_copy: ViT
    opt: torch.optim.Optimizer
    grad_scaler: torch.cuda.amp.GradScaler
    config: Any
    
    def __post_init__(self):
        self.device = self.config.train.device
        if self.config.train.distributed:
            self.f = nn.DataParallel(self.f)
            self.f_copy = nn.DataParallel(self.f_copy)
            
        self.f.to(self.device)
        self.f_copy.to(self.device)
        self.global_step = 0
        
    def get_states(self):
        try:
            f_state_dict = self.f.module.state_dict()
        except AttributeError:
            f_state_dict = self.f.state_dict()
        
        return dict(
            f=f_state_dict,
            opt=self.opt.state_dict(),
            grad_scaler=self.grad_scaler.state_dict(),
            config=self.config.model
        )
    
    def update(self):
        self.opt.zero_grad()
        self.grad_scaler.scale(self.loss).backward()
        self.grad_scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.f.parameters(), max_norm=1.0)
        self.grad_scaler.step(self.opt)
        self.grad_scaler.update()
        self.global_step += 1
    
    def train_step(self, x):
        self.f.train()
        self.f_copy.train()
        
        with torch.autocast(self.device, torch.float16, self.config.train.use_amp):
            z = torch.randn_like(x)
            self.f_copy.load_state_dict(self.f.state_dict())
            fx = self.f(x)
            fz = self.f(z)
            f_z = fz.detach()
            ff_z = self.f(f_z)
            f_fz = self.f_copy(fz)

            self.loss_rec = (fx - x).pow(2).mean()
            self.loss_idem = (f_fz - fz).pow(2).mean()
            self.loss_tight = -(ff_z - f_z).pow(2).mean()
            
            self.loss = self.loss_rec + self.loss_idem + self.loss_tight * 0.1
        self.update()
    
    def metric_info(self):
        return dict(
            loss=self.loss.detach().item(),
            loss_rec=self.loss_rec.detach().item(),
            loss_idem=self.loss_idem.detach().item(),
            loss_tight=self.loss_tight.detach().item()
        )
        
    def train(self, data_loader, logger: Logger):
        n_epochs = self.config.train.n_epochs
        for epoch in range(1, 1 + n_epochs):
            for x in (bar := tqdm(data_loader)):
                x = x.to(self.device)
                self.train_step(x)
                
                logger.log(global_step=self.global_step, **self.metric_info())
                bar.set_description(f'epoch {epoch}')
                bar.set_postfix(**self.metric_info())
        imgs = sample(self.f, 16, self.config)
        logger.log(images=imgs)
        