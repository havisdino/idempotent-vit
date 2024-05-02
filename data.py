from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os


def img2seq(images, d_patch):
    B = images.size(0)
    return images.view(B, -1, d_patch)


def seq2img(sequences, img_shape):
    B = sequences.size(0)
    return sequences.view(B, *img_shape)


class ImageDataset(Dataset):
    def __init__(self, directory, img_size, d_patch):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x * 2. - 1),
            T.Resize(img_size, antialias=True),
            T.Lambda(lambda x: img2seq(x, d_patch))
        ])
        
        self.directory = directory
        self.fnames = os.listdir(directory)

    def get_path(self, idx):
        fname = self.fnames[idx]
        return os.path.join(self.directory, fname)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = self.get_path(idx)
        img = Image.open(path)
        img = self.transform(img)
        return img
