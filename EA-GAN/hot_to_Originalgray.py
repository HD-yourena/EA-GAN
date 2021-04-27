# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:25:21 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:52:58 2020

@author: Administrator
"""
import numpy as np
import cv2
    
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize
from PIL import Image
from math import log10
import pytorch_ssim

def hot_to_Originalgray(epoch,method):
    inpath='F:/contrast/contrast_vis/%d' %(epoch)+method+'.png'
    img = cv2.imread(inpath)
#    imgGrey = cv2.imread(inpath, 0)
    b , g , r = cv2.split(img)
    B = b.astype(np.float32) / 255
    G = g.astype(np.float32) / 255
    R = r.astype(np.float32) / 255
    Gr=63*(1-B)+96*(2-G-R)
    Gr=Gr.astype(np.uint8)
    outpath_o='F:/contrast/contrast_vis_o/%d_o_' %(epoch)+method+'.png'
    cv2.imwrite(outpath_o,np.uint8(Gr))
    
    x = cv2.Sobel(Gr,cv2.CV_16S,1,0)
    y = cv2.Sobel(Gr,cv2.CV_16S,0,1)
     
    absX = cv2.convertScaleAbs(x)# 转回uint8
    absY = cv2.convertScaleAbs(y)
     
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)


    outpath_osr='F:/contrast/contrast_vis_o/%d_osr_' %(epoch)+method+'.png'
#    cv2.imwrite(outpath_osr,np.uint8(gaussImg))
    cv2.imwrite(outpath_osr,dst) 



def jisuan(epoch):
    ea_wgan=Image.open('F:/contrast/contrast_vis_o/%d_osr_' %(epoch)+'ea_wgan'+'.png')
    hr=Image.open('F:/contrast/contrast_vis_o/%d_osr_' %(epoch)+'hr'+'.png')
    wgan=Image.open('F:/contrast/contrast_vis_o/%d_osr_' %(epoch)+'wgan'+'.png')
    bic=Image.open('F:/contrast/contrast_vis_o/%d_osr_' %(epoch)+'bic'+'.png')
    norm = ToTensor()
    
    ea_wgan=norm(ea_wgan)
    hr=norm(hr)
    wgan=norm(wgan)
    bic=norm(bic)
    psnr_eawgan=10 * log10(1 / ((ea_wgan - hr) ** 2).mean().item())
    psnr_wgan=10 * log10(1 / ((wgan - hr) ** 2).mean().item())
    psnr_bic=10 * log10(1 / ((bic - hr) ** 2).mean().item())
    ssim_eawgan = pytorch_ssim.ssim(ea_wgan.unsqueeze(0), hr.unsqueeze(0)).item()
    ssim_wgan = pytorch_ssim.ssim(wgan.unsqueeze(0), hr.unsqueeze(0)).item()
    ssim_bic = pytorch_ssim.ssim(bic.unsqueeze(0), hr.unsqueeze(0)).item()
    #cv2.imshow("c", np.uint8(dSobel))
    return psnr_eawgan,psnr_wgan,psnr_bic,ssim_eawgan,ssim_wgan,ssim_bic
out_size=4#outpath_osr中比较的图像数量
for epoch in range(out_size):
    for method in ['_lr','bic','wgan','ea_wgan','hr']:
        hot_to_Originalgray(epoch+1,method)    
    



out=[]
for i in range(outsize):
    out.extend([jisuan(i+1)])    
def mean_jisuan(i):
    return np.mean([jisuan(1)[i],jisuan(2)[i],jisuan(3)[i],jisuan(4)[i]])
    
    
    
    
    
    