
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
with open('/path_to/semanticCNN/imagenet_labels/ILSVRC2012_mapping.txt') as f:
     label_map = f.read()
     for line in label_map.splitlines():
         fields = line.strip().split()
         mapping_table[fields[0]] = fields[1]
def get_annotations_map():
    valAnnotationsPath = '/path_to/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
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
img_dir = '/path_to/ImageNet_Val/'
# Specify image dimensions
size = 224


# Load images
val_annotations_map = get_annotations_map()
all_images = []
all_annotations = []
i = 0






for filename in os.listdir(img_dir):
    if not filename.startswith('.'):
        index = (int)(filename.split(".JPEG")[0].split("_")[2])
        #print(index)
        img = image.load_img(img_dir + filename, target_size = (size, size)) # We assume all images have the same dimensions
        img = image.img_to_array(img)
        all_images.append(img)
        all_annotations.append(val_annotations_map[index-1])
'''
# Display images
for key, vals in all_images.items():
    fig2 = plt.figure()
    plt.axis('off')
    plt.imshow(vals.astype(np.uint8))
'''

# Load model
model = VGG19(weights = 'imagenet')

def PredictSimplexRand(images, labels, max_norm, freq_sine, adversarial = False):
    size = 224
    j = 0
    FEATURE_SIZE = 40.0


    simplex = OpenSimplex()

    print('Generating 2D image...')
    noise_color = []
    noise = Image.new('L', (size, size))
    for y in range(0, size):
        for x in range(0, size):
            value = simplex.noise2d(x / FEATURE_SIZE, y / FEATURE_SIZE)
            color = int((value + 1) * 128)
            noise.putpixel((x, y), color)
    noise = normalize(np.array(noise))
    noise = np.sin(noise * freq_sine * np.pi)
    noise_color = colorize(normalize(noise))  
    for i in range(len(labels)):
        image = images[i]
        #octave_value = random.randint(1, 4)
        #noise = perlin(size = size, period = period, octave = octave_value, freq_sine = freq_sine)
        #noise = colorize(noise)
        if adversarial == False:
            payload = perturb(img = image, noise = np.zeros((size, size, 3)), norm = max_norm)
        else:
            payload = perturb(img = image, noise = noise_color, norm = max_norm)
        payload = payload.astype('float32')
        payload = cv2.bilateralFilter(payload, 3, 5, 5)        
        prob = model.predict(preprocess_input(payload.astype(np.float).reshape((1, size, size, 3))))
        index = (labels[i]-1)
        label_str_list = decode_predictions(prob, top =1)[0]
        label_str = ""
        for item in label_str_list:
            print(item)
            label_str = item[0]
        print("The prediction is %s" %label_str) 
        ground_truth_str = mapping_table[str(labels[i])]
        print("The ground truth is %s" %ground_truth_str)
        if ground_truth_str != label_str:
            j = j + 1
            print("Not equal")
        else:
            print("Equal")
        real_attack_rate = (float)(j)/(float)(i+1)
        print('%.4f' %real_attack_rate)
    evasion_rate = (float)(j) / (float)(len(labels)) 
    return evasion_rate



# Parameter sliders
#s_img_key = Dropdown(options = list(all_images.keys()), value = 'boat', description = 'Image:')
'''
s_max_norm = IntSlider(min = 0, max = 64, value = 12, step = 2, continuous_update = False, description = 'Max Change:')
s_num_kern = IntSlider(min = 1, max = 100, value = 23, step = 1, continuous_update = False, description = 'No. Kernels:')
s_ksize = IntSlider(min = 1, max = 100, value = 23, step = 1, continuous_update = False, description = 'Kernel Size:')
s_sigma = FloatSlider(min = 1, max = 20, value = 8, step = 0.25, continuous_update = False, description = 'Kernel Var:')
s_theta = FloatSlider(min = 0, max = np.pi, value = np.pi / 4, step = np.pi / 24, continuous_update = False, description = 'Orientation:')
s_lambd = FloatSlider(min = 0.25, max = 20, value = 8, step = 0.25, continuous_update = False, description = 'Bandwidth:')
s_color = ToggleButtons(options = ['Black-White', 'Red-Cyan', 'Green-Magenta', 'Blue-Yellow'], description = 'Color:', button_style='', disabled = False)
'''

s_max_norm = 12
#s_num_kern = 23
#s_ksize = 23
#s_sigma = 8.00
#s_theta = 0.79
#s_lambd = 8.00
#s_color = 'Black-White' 
s_freq_sine = 36
input_specific_evasion_rate =PredictSimplexRand(
                 all_images,
                 all_annotations,
                 max_norm = s_max_norm,
                 freq_sine = s_freq_sine,
                 adversarial = True)
print('Attack effect on vgg19: %.4f' %input_specific_evasion_rate)
