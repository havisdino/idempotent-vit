import torch
from modules import ViT
from data import seq2img


@torch.no_grad()
def sample(f: ViT, n_samples, config):
    z_shape = [config.model.n_patches, config.model.d_patch]
    img_shape = config.data.img_shape
    device = config.train.device
    
    f.eval()
    z = torch.randn(n_samples, *z_shape, device=device)
    x = f(z)
    x = x.cpu()
    return seq2img(x, img_shape)
    