import os
import argparse
import time

from tqdm import tqdm
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pytorch_ssim
import torchvision.utils as utils
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from math import log10
import pandas as pd

from val_load import DevDataset, to_image
from model import Generator as GWGAN
from model_EA import Generator_EAGAN as GEAWGAN
cache = {'psnr_bic': [], 'ssim_bic': [],'psnr_WGAN': [], 'ssim_WGAN': [], 'psnr_EAWGAN': [], 'ssim_EAWGAN': [], 'contrast_psnr_wgan':[], 'contrast_ssim_wgan':[],'contrast_psnr_bic':[],'contrast_ssim_bic':[]}
def main():
	parser = argparse.ArgumentParser(description='Validate SRGAN')
	parser.add_argument('--val_set', default='F:/hot_test', type=str, help='dev set path')
	parser.add_argument('--m0', default='F:/contrast/WGAN_hot/netG_epoch_495_gpu.pth', type=str, help='model0')
	parser.add_argument('--m1', default='F:/contrast/EAWGAN_hot/netG_epoch_495_gpu.pth', type=str, help='model1')
	
	opt = parser.parse_args()
	val_path = opt.val_set
	m0 = opt.m0
	m1 = opt.m1

	val_set = DevDataset(val_path, upscale_factor=4)
	val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
	
#	now = time.gmtime(time.time())
	#configure(str(now.tm_mon) + '-' + str(now.tm_mday) + '-' + str(now.tm_hour) + '-' + str(now.tm_min), flush_secs=5)
        
	netGWGAN = GWGAN()
	netGEAWGAN = GEAWGAN()

	if torch.cuda.is_available():
		netGWGAN.cuda()
		netGEAWGAN.cuda()	
        
	out_path = 'F:/contrast/contrast_vis/'
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	with torch.no_grad():
		netGWGAN.eval()
		netGEAWGAN.eval()

		val_bar = tqdm(val_loader)
		dev_images = []
		dev_low = []
		
		for val_lr, val_bic, val_hr in val_bar:
#			batch_size = val_lr.size(0)

			if torch.cuda.is_available():
				lr = val_lr.cuda()
				hr = val_hr.cuda()
				bic = val_bic.cuda()
			netGWGAN.load_state_dict(torch.load(m0))	
			sr0 = netGWGAN(lr)
			
			netGEAWGAN.load_state_dict(torch.load(m1))	
			sr1 = netGEAWGAN(lr)
			
            
            
			psnr_bic = 10 * log10(1 / ((bic - hr) ** 2).mean().item())
			ssim_bic = pytorch_ssim.ssim(bic, hr).item()            
			psnr_WGAN = 10 * log10(1 / ((sr0 - hr) ** 2).mean().item())
			ssim_WGAN = pytorch_ssim.ssim(sr0, hr).item()
			psnr_EAWGAN = 10 * log10(1 / ((sr1 - hr) ** 2).mean().item())
			ssim_EAWGAN = pytorch_ssim.ssim(sr1, hr).item()
			cache['psnr_bic'].extend([psnr_bic])
			cache['ssim_bic'].extend([ssim_bic])            
			cache['psnr_WGAN'].extend([psnr_WGAN])
			cache['ssim_WGAN'].extend([ssim_WGAN])
			cache['psnr_EAWGAN'].extend([psnr_EAWGAN])
			cache['ssim_EAWGAN'].extend([ssim_EAWGAN])
			cache['contrast_psnr_wgan'].extend([psnr_EAWGAN-psnr_WGAN])            
			cache['contrast_ssim_wgan'].extend([ssim_EAWGAN-ssim_WGAN])            
			cache['contrast_psnr_bic'].extend([psnr_EAWGAN-psnr_bic])            
			cache['contrast_ssim_bic'].extend([ssim_EAWGAN-ssim_bic])               
            
            
#			netG.load_state_dict(torch.load('cp/netG_baseline_gpu.pth'))
#			sr_baseline = netG(lr)
			
			# Avoid out of memory crash on 8G GPU
			if len(dev_images) <  400:
				dev_low.extend([to_image()(lr.data.cpu().squeeze(0))])
				dev_images.extend([to_image()(bic.data.cpu().squeeze(0)), to_image()(sr0.data.cpu().squeeze(0)), to_image()(sr1.data.cpu().squeeze(0)), to_image()(hr.data.cpu().squeeze(0))])
#to_image()(sr_baseline.data.cpu().squeeze(0)),	
                
		dev_low = torch.stack(dev_low)
		dev_images = torch.stack(dev_images)
#		dev_images = torch.chunk(dev_images, dev_images.size(0) // 5)

		low_save_bar = tqdm(dev_low, desc='[saving images]')
		dev_save_bar = tqdm(dev_images, desc='[saving images]')
		index = 1
		Index = 1        
        
		for image in low_save_bar:
			image = utils.make_grid(image, nrow=1, padding=1)
			utils.save_image(image, out_path + '%d_lr.png' % (index), padding=1)
			index += 1        
		for image in dev_save_bar:
			name=['bic','wgan','ea_wgan','hr'] 
			namedex=(Index-1)%4
			image = utils.make_grid(image, nrow=4, padding=1)
			utils.save_image(image, out_path + '%d'% ((Index-namedex+3)/4)+name[namedex]+'.png', padding=1)
			Index += 1
			
if __name__ == '__main__':
	main()
mean = {'psnr_bic': 0, 'ssim_bic': 0,'psnr_WGAN': 0, 'ssim_WGAN': 0, 'psnr_EAWGAN': 0, 'ssim_EAWGAN': 0, 'contrast_psnr_wgan':0, 'contrast_ssim_wgan':0,'contrast_psnr_bic':0,'contrast_ssim_bic':0}
for i in mean:
    mean[i]=np.mean(cache[i])