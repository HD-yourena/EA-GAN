import argparse
import time
import os

from os.path import basename, normpath

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model_EA import Generator_EAGAN

parser = argparse.ArgumentParser(description='SR single image')
parser.add_argument('--lr', default='',type=str, help='C:/Users/Administrator/Desktop/rcx/a.jpg')
parser.add_argument('--m', default='', type=str, help='model')
opt = parser.parse_args()

lr = opt.lr
pth = opt.m
with torch.no_grad():
	sr_path = ''
	if not os.path.exists(sr_path):
		os.makedirs(sr_path)
		
	model = Generator_EAGAN().eval()
	if torch.cuda.is_available():
		model.cuda()
	model.load_state_dict(torch.load(pth))

	image = Image.open(lr)

	image = Variable(ToTensor()(image)).unsqueeze(0)
	
	if torch.cuda.is_available():
		image = image.cuda()

	out = model(image)
	out_img = ToPILImage()(out[0].data.cpu())
	out_img.save(sr_path +"WGAN_"+ basename(normpath(lr)))