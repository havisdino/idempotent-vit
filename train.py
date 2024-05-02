import torch
from torch.utils.data import DataLoader

from data import ImageDataset
from logger import TensorBoardLogger
from modules import get_model_from_config
from trainer import Trainer
from utils import load_config


def main():
    config = load_config()
    logger = TensorBoardLogger()

    f = get_model_from_config(config.model)
    f_copy = get_model_from_config(config.model)
    opt = torch.optim.Adam(f.parameters(), config.train.learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler()

    trainer = Trainer(f, f_copy, opt, grad_scaler, config)
    dataset = ImageDataset(config.train.data_path, config.data.img_shape[1:], config.model.d_patch)
    data_loader = DataLoader(
        dataset, config.train.batch_size,
        shuffle=True, prefetch_factor=2, num_workers=2
    )
    trainer.train(data_loader, logger)


if __name__ == '__main__':
    main()