# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:34:57 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:44:21 2020

@author: Administrator
"""
from PIL import Image,ImageDraw
def imageResize(input_path, scale):
    # 获取输入文件夹中的所有文件/夹，并改变工作空间
    #files = os.listdir(input_path)
    #os.chdir(input_path)
    # 判断输出文件夹是否存在，不存在则创建
    #if(not os.path.exists(output_path)):
    #    os.makedirs(output_path)
    #for file in files:
        # 判断是否为文件，文件夹不操作
        #if(os.path.isfile(file)):
    img = input_path
    width = int(img.size[0]*scale)
    height = int(img.size[1]*scale)
    img = img.resize((width, height), Image.ANTIALIAS)
    return img
def Partial_zoom(epoch,method_epoch,center):
    method=['_lr','bic','wgan','ea_wgan','hr']
    in_path='F:/contrast/contrast_vis/%d' %(epoch)
    out_path='F:/contrast/Partial_zoom/%dPZ' %(epoch)
    NAME=method[method_epoch]+'.png'
    out_image=out_path+NAME
    IMAGE_NAME=in_path+NAME
    #image=image.resize((),Image.ANTIALIAS)
    #image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    img = Image.open(IMAGE_NAME)
    # 图片尺寸
    img_size = img.size
    h0 = img_size[1]  # 图片高度
    w0 = img_size[0]  # 图片宽度
     
    x = (center[0]-0.1) * w0
    y = (center[1]-0.1) * h0
    w = (center[0]+0.1) * w0
    h = (center[1]+0.1) * h0
     
    # 开始截取
    region = img.crop((x, y, w, h))
    # 保存图片
    Scale=2
    UP_region=imageResize(region,Scale)
    draw =ImageDraw.Draw(img)
    if method[method_epoch]=='_lr':
        line=2
    else:
        line=5
    
    for i in range(1, line):
        draw.rectangle((x-(line - i), y-(line - i),w+(line - i), h+(line - i)), fill=None, outline='white')
    #for p in points:
    #    draw.ellipse([p[0],p[1],p[2],p[3]], fill='red')
    X=int(Scale*region.size[0])
    Y=int(Scale*region.size[1])
    img.paste(UP_region,(0,0,X,Y))
    for i in range(1, line):
        draw.rectangle((0-(line - i), 0-(line - i),X+(line - i), Y+(line - i)), fill=None, outline='white')
    img = img.convert('RGB')
    img.save(out_image)
def PZ(epoch,center):
     for i in range(5):
         Partial_zoom(epoch,i,center)