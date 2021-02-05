
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from ipywidgets import interactive
from ipywidgets import Dropdown, FloatSlider, IntSlider, ToggleButtons
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import decode_predictions, preprocess_input
from keras.preprocessing import image
from utils_attack import colorize, perturb
from utils_noise import perlin
from utils_noise import normalize, normalize_var
import json
from PIL import Image # Depends on the Pillow lib
import cv2
from opensimplex import OpenSimplex
mapping_table = {}
#https://github.com/fh295/semanticCNN
#https://github.com/0429charlie/ImageNet_metadata
with open('/root/junyan/adversarial_examples/semanticCNN/imagenet_labels/ILSVRC2012_mapping.txt') as f:
     label_map = f.read()
     for line in label_map.splitlines():
         fields = line.strip().split()
         mapping_table[fields[0]] = fields[1]
def get_annotations_map():
    valAnnotationsPath = '/root/junyan/adversarial_examples/ImageNet_metadata/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = []

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        #valAnnotations[pieces[0]] = pieces[1]
        data_value = (int)(pieces[0])
        #print(data_value)
        #if data_value == 0:
        #    print("Get zero")
        #if data_value == 1000:
        #    print("Get thousand")
        valAnnotations.append(data_value)

    return valAnnotations
#img_dir = 'images/'
img_dir = '/root/junyan/ImageNet_Val/'
# Specify image dimensions
size = 224


# Load images
val_annotations_map = get_annotations_map()
#all_images = []
#all_annotations = []
i = 0





model = VGG19(weights = 'imagenet')
size = 224
j = 0
FEATURE_SIZE = 40.0

simplex = OpenSimplex()

print('Generating 2D image...')
noise_color = []
freq_sine = 36
noise = Image.new('L', (size, size))
for y in range(0, size):
    for x in range(0, size):
        value = simplex.noise4d(x / FEATURE_SIZE, y / FEATURE_SIZE, 0.0)
        color = int((value + 1) * 128)
        noise.putpixel((x, y), color)
noise = normalize(np.array(noise))
noise = np.sin(noise * freq_sine * np.pi)
noise_color = colorize(normalize(noise))  
max_norm = 12
i = 0 
for filename in os.listdir(img_dir):
    if not filename.startswith('.'):
        index = (int)(filename.split(".JPEG")[0].split("_")[2])
        #print(index)
        img = image.load_img(img_dir + filename, target_size = (size, size)) # We assume all images have the same dimensions
        img = image.img_to_array(img)
        label = val_annotations_map[index-1]
        i = i + 1
        payload = perturb(img = img, noise = noise_color, norm = max_norm)
        payload = payload.astype('float32')
        payload = cv2.bilateralFilter(payload, 3, 5, 5)        
        prob = model.predict(preprocess_input(payload.astype(np.float).reshape((1, size, size, 3))))
        index = (label-1)
        label_str_list = decode_predictions(prob, top =1)[0]
        label_str = ""
        for item in label_str_list:
            print(item)
            label_str = item[0]
        print("The prediction is %s" %label_str) 
        ground_truth_str = mapping_table[str(label)]
        print("The ground truth is %s" %ground_truth_str)
        if ground_truth_str != label_str:
            j = j + 1
            print("Not equal")
        else:
            print("Equal")
        real_attack_rate = (float)(j)/(float)(i)
        print('%.4f' %real_attack_rate)
last_attack_rate = (float)(j)/(float)(i)
print('Last attack rate on VGG 19 is %.4f' %last_attack_rate) 


