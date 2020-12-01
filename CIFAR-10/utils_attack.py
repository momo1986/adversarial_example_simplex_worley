from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

'''
Color noise

noise           has dimension 2 or 3, pixel range [0, 255]
color            is [a, b, c] where a, b, c are from {-1, 0, 1}
'''
def colorize(noise, color = [1, 1, 1]):
    if noise.ndim == 2: # expand to include color channels
        noise = np.expand_dims(noise, 2)
    return (noise - 0.5) * color * 2 # output pixel range [-1, 1]

'''
Perturb image and clip to maximum perturbation norm

img              image with pixel range [0, 1]
noise           noise with pixel range [-1, 1]
norm           L-infinity norm constraint
'''
def perturb(img, noise, norm):
    noise = np.sign((noise - 0.5) * 2) * norm
    noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
    return (img + noise)

# adds the gaussian noise based on the mean and the standard deviation 
def add_gaussian_noise(img):
  mean = (10, 10, 10)
  std = (50, 50, 50)
  row, col, channel = img.shape
  noise = np.random.normal(mean, std, (row, col, channel)).astype('uint8')
  return img + noise

def perturb_gaussian_noise(img, norm):
  mean = (10, 10, 10)
  std = (50, 50, 50)
  row, col, channel = img.shape
  noise = np.random.normal(mean, std, (row, col, channel)).astype('uint8')
  noise = np.sign((noise - 0.5) * 2) * norm
  noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
  return img + noise

def sp_noise(image,prob=0.1):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def perturb_sp_noise(img, norm, prob=0.1):
  noise = sp_noise(img, prob)
  noise = np.sign((noise - 0.5) * 2) * norm
  noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
  return img + noise
