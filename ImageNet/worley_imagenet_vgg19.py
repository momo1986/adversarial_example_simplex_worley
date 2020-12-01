
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
import math
import sys
from random import randint
import cv2
mapping_table = {}

#https://github.com/fh295/semanticCNN
#https://github.com/0429charlie/ImageNet_metadata//hub.baai.ac.cn/view/4498
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

def create_img_grid(img_width, img_height):
    grid = []
    for i in range(img_height):
        grid_row = []
        for j in range(img_width):
            grid_row.append(None)

        grid.append(grid_row)

    return grid

def select_random_points(img_width, img_height, num_points):
    max_width = img_width - 1
    max_height = img_height - 1
    random_points = []
    for i in range(num_points):
        random_points.append(
            tuple([ randint(0, max_width), randint(0, max_height) ])
        )

    return random_points

def get_normalized_distance_from_nearest_point(pixel_x,
                                                pixel_y,
                                                img_width,
                                                img_height,
                                                random_points):
    shortest_norm_dist = 1
    for point_x, point_y in random_points:
        x_dist = (pixel_x - point_x) / (img_width / 4)
        y_dist = (pixel_y - point_y) / (img_height / 4)
        norm_dist = math.sqrt(x_dist ** 2 + y_dist ** 2)

        shortest_norm_dist = min(norm_dist, shortest_norm_dist)

    return shortest_norm_dist



def map_to_bw_colour(colour_val):
    # We should return this as inverted so that pixels nearer to a selected
    # point is closer to a white colour.
    color_val = (float)(colour_val)
    return (1 - colour_val) * 255


def clear_line_printed():
    global len_prev_printed_line
    print(f'\r{" " * len_prev_printed_line}\r', end='', flush=True)

def one_line_print(text: str):
    global len_prev_printed_line
    clear_line_printed()
    print(f'{text}', end='', flush=True)

    len_prev_printed_line = len(text)

is_verbose = False
is_one_line_output = True
show_starting_points =  True
len_prev_printed_line = 0




def PredictWorleyRand(images, labels, max_norm, freq_sine, adversarial = False):
    size = 224
    j = 0
    NUM_POINTS = 100
    img_grid = create_img_grid(size, size)
    rand_points = select_random_points(size, size, NUM_POINTS)
    for y in range(size):
        for x in range(size):
            bw_colour = map_to_bw_colour(
                            get_normalized_distance_from_nearest_point(
                                x, y, size, size, rand_points))
            # NOTE: Colour is in RGBA.
            img_grid[y][x] = tuple(([ round(bw_colour) ] * 3) + [ 255 ])
    if show_starting_points:
        print('Random Points')
        print('-------------')
        for x, y in rand_points:
            print(f'({x},\t{y})')
            img_grid[y][x] = tuple([ 255, 0, 0, 255 ])
    noise = np.array(img_grid)
    b,g,r,d = cv2.split(noise)
    b = np.expand_dims(b, axis =2)
    g = np.expand_dims(g, axis=2)
    r = np.expand_dims(r, axis=2)
    print(b.shape)
    noise = np.concatenate([b, g, r], axis=2)
    print(noise.shape)
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
input_specific_evasion_rate =PredictWorleyRand(
                 all_images,
                 all_annotations,
                 max_norm = s_max_norm,
                 freq_sine = s_freq_sine,
                 adversarial = True)
print('Attack effect on VGG-19: %.4f' %input_specific_evasion_rate)
