#!/usr/bin/python
# encoding: utf-8

import random
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
from PIL import Image,ImageEnhance,ImageOps
import numpy as np
import trans
import config

debug_idx = 0
debug = True

#crop = trans.Crop(probability=0.1)
crop2 = trans.Crop2(probability=1.1)
random_contrast = trans.RandomContrast(probability=0.1)
random_brightness = trans.RandomBrightness(probability=0.1)
random_color = trans.RandomColor(probability=0.1)
random_sharpness = trans.RandomSharpness(probability=0.1)
compress = trans.Compress(probability=0.3)
exposure = trans.Exposure(probability=0.1)
rotate = trans.Rotate(probability=0.1)
blur = trans.Blur(probability=0.1)
salt = trans.Salt(probability=0.1)
adjust_resolution = trans.AdjustResolution(probability=0.1)
stretch = trans.Stretch(probability=0.1)

#crop.Crop2(probability=1.1)
crop2.setparam()
random_contrast.setparam()
random_brightness.setparam()
random_color.setparam()
random_sharpness.setparam()
compress.setparam()
exposure.setparam()
rotate.setparam()
blur.setparam()
salt.setparam()
adjust_resolution.setparam()
stretch.setparam()

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint( 0, 31 ) / 10.  # 随机因子
    color_image = ImageEnhance.Color( image ).enhance( random_factor )  # 调整图像的饱和度
    random_factor = np.random.randint( 10, 21 ) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness( color_image ).enhance( random_factor )  # 调整图像的亮度
    random_factor = np.random.randint( 10, 21 ) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast( brightness_image ).enhance( random_factor )  # 调整图像对比度
    random_factor = np.random.randint( 0, 31 ) / 10.  # 随机因子
    return ImageEnhance.Sharpness( contrast_image ).enhance( random_factor )  # 调整图像锐度

def randomGaussian(image, mean=0.2, sigma=0.3):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range( len( im ) ):
            im[_i] += random.gauss( mean, sigma )
        return im

    # 将图像转化成数组
    img = np.asarray( image )
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy( img[:, :, 0].flatten(), mean, sigma )
    img_g = gaussianNoisy( img[:, :, 1].flatten(), mean, sigma )
    img_b = gaussianNoisy( img[:, :, 2].flatten(), mean, sigma )
    img[:, :, 0] = img_r.reshape( [width, height] )
    img[:, :, 1] = img_g.reshape( [width, height] )
    img[:, :, 2] = img_b.reshape( [width, height] )
    return Image.fromarray( np.uint8( img ) )

def inverse_color(image):
    if np.random.random()<0.4:
        image = ImageOps.invert(image)
    return image


def data_tf_fullimg(img,loc):
    # 从整图中截取文字块并进行处理
    left, top, right, bottom = loc
    img = crop2.process([img, left, top, right, bottom])
    img = random_contrast.process(img)
    img = random_brightness.process(img)
    img = random_color.process(img)
    img = random_sharpness.process(img)
    img = compress.process(img)
    img = exposure.process(img)
    # img = rotate.process(img)
    img = blur.process(img)
    img = salt.process(img)
    # img = inverse_color(img)
    # img = adjust_resolution.process(img)
    # img = stretch.process(img)
    return img



class MyDataset(Dataset):
    def __init__(self, txt_folder = './train_data/gt/', img_folder = './train_data/img/', train=True, fullimg_transform = data_tf_fullimg, target_transform = None):
        super(Dataset, self).__init__()
        self.fullimg_transform = fullimg_transform
        self.target_transform = target_transform
        self.TXTs = os.listdir(txt_folder) # 训练集对应所有TXT名称
        # self.TXTs = ['tr_img_XXXXX.txt', 'tr_img_XXXXX.txt, ...]
        self.train = train
        self.imgs = list()
# =============================================================================
#         self.files = list()
#         self.labels = list() # 每个文字块的实际内容
#         self.locs = list() # 每个文字块在全图中的位置
# 
# =============================================================================
        for TXT in self.TXTs:
            fname = os.path.join(img_folder, TXT.replace('.txt', '.jpg'))
            TXT_file = os.path.join(txt_folder, TXT)
            with open(TXT_file) as f:
                content = f.readlines()
                for line in content:
                    w1, h1, w2, h2, w3, h3, w4, h4, script, label = line.strip().split(',', 9)
                    if set(label) == set('#'):
                        continue
                    left = max(min(int(w1), int(w2), int(w3), int(w4)), 0)# left 至少是0
                    right = max(1, int(w1), int(w2), int(w3), int(w4))# right 至少是1
                    top = max(min(int(h1), int(h2), int(h3), int(h4)), 0)# top 至少是0
                    bottom = max(1, int(h1), int(h2), int(h3), int(h4))# bottom 至少1
                    loc = [left, top, right, bottom]
                    self.imgs.append((fname, label, loc))
# =============================================================================
#                     self.files.append(fname)
#                     self.labels.append(label)
#                     self.locs.append([min(int(left1), int(left2)), min(int(top1), int(top2)),
#                                       max(int(right1), int(right2)), max(int(bottom1), int(bottom2))])
#                     # locs = [[left, top, right, bottom], [left, top, right, bottom], ...]
# =============================================================================
        
        self.n = len(self.imgs) # 文字块总数
        
    def name(self):
        return 'MyDatasetPro'

    def __getitem__(self, index):
#        label = self.labels[index]
        label = self.imgs[index][1]
        if self.target_transform is not None:
            label = self.target_transform(label)
        img = Image.open(self.imgs[index][0])
        img = self.fullimg_transform(img, self.imgs[index][2])
#        img = Image.open(self.files[index])
#        img = self.fullimg_transform(img, self.locs[index])
        img = img.convert('L') # 灰度显示
        return (img,label)

    def __len__(self):
        return self.n


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img): # 保证h是固定的imgH，w不少于imgW
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w<=(w0/h0*h): # 图片weight超出，则需reshape到w
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5) # (tensor - 0.5) / 0.5 就地修改
        else:
            w_real = max(int(w0/h0*h), 1) # 图片weight不足，不需处理
            img = img.resize((w_real,h), self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5) # (tensor - 0.5) / 0.5 就地修改
            start = random.randint(0,w-w_real-1)
# =============================================================================
#             if self.is_test: # 测试集图片统一更改为[(5个像素)，(测试图片)，(5个像素)]
#                 start = 5
#                 w += 10
# =============================================================================
            tmp = torch.zeros([img.shape[0], h, w])+0.5 # 设置背景色
            tmp[:,:,start:start+w_real] = img # 背景图里随机选择一个位置替换为img
            img = tmp #更新image实现输入图片维度一致
        return img

class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index) # 生成迭代器

    def __len__(self):
        return self.num_samples

class alignCollate(object):

    def __init__(self, imgH=config.imgH, imgW=config.imgW, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = list(zip(*batch))

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels



if __name__ == '__main__':
    path = 'train_data/gt'
    files = os.listdir(path)
    idx = 0
    for f in files:
        img_name = os.path.join(path,f)
        img = Image.open(img_name)
        img.show()
        idx+=1
        if idx>5:
            break