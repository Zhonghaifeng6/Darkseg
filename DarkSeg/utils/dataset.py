
import os
import os.path as osp
import logging
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, dark_dir):
        self.imgs_dir   = imgs_dir
        self.masks_dir  = masks_dir
        self.dark_dir   = dark_dir

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img):
        # Resize all images to 256x256
        pil_img = pil_img.resize((256, 256))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            # For grayscale mask, add channel dimension
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # For RGB or grayscale image, normalize
            img_nd = img_nd / 255

        # Convert HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)
        dark_path = osp.join(self.dark_dir, img_name)
        mask_path = osp.join(self.masks_dir, img_name)

        img = Image.open(img_path)
        dark = Image.open(dark_path)
        mask = Image.open(mask_path)

        img = self.preprocess(img)
        dark = self.preprocess(dark)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(img).float(),
            'dark': torch.from_numpy(dark).float(),
            'mask': torch.from_numpy(mask).float()
        }



