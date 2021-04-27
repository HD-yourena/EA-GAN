import numpy as np
import torch

import os
from os import listdir
from os.path import join

from PIL import Image

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def to_image():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])       
    
    

class DevDataset(Dataset):
	def __init__(self, dataset_dir, upscale_factor,):
		super(DevDataset, self).__init__()
		self.upscale_factor = upscale_factor
		self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

	def __getitem__(self, index):
		hr_image = Image.open(self.image_filenames[index])
		crop_size = calculate_valid_crop_size(400, self.upscale_factor)
		lr_scale = Resize(crop_size//self.upscale_factor, interpolation=Image.BICUBIC)
		hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
		hr_image = CenterCrop(crop_size)(hr_image)
		lr_image = lr_scale(hr_image)
		hr_restore_img = hr_scale(lr_image)
		norm = ToTensor()
		return norm(lr_image), norm(hr_restore_img), norm(hr_image)

	def __len__(self):
		return len(self.image_filenames)

