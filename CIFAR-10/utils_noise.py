from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2

from random import randint
import math
### Helper Functions ###
'''
Normalize variance spectrum

Implementation based on https://hal.inria.fr/hal-01349134/document 
Fabrice Neyret, Eric Heitz. Understanding and controlling contrast oscillations in stochastic texture
algorithms using Spectrum of Variance. [Research Report] LJK / Grenoble University - INRIA. 2016,
pp.8. <hal-01349134>
'''
def normalize_var(orig):
    size = orig.shape[0]
    
    # Spectral variance
    mean = np.mean(orig)
    spec_var = np.fft.fft2(np.square(orig -  mean))
    
    # Normalization
    imC = np.sqrt(abs(np.real(np.fft.ifft2(spec_var))))
    imC /= np.max(imC)
    minC = 0.001
    imK =  (minC + 1) / (minC + imC)
    
    img = mean + (orig -  mean) * imK    
    return normalize(img)

# Normalize vector
def normalize(vec):
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)

# Valid positions for Gabor noise
def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True


### Procedural Noise ###
# Note: Do not take these as optimized implementations.
'''
Gabor kernel

sigma       variance of gaussian envelope
theta         orientation
lambd       sinusoid wavelength, bandwidth
xy_ratio    value of x/y
psi            phase shift of cosine in kernel
sides        number of directions
'''
def gaborK(ksize, sigma, theta, lambd, xy_ratio, sides):
    gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    for i in range(1, sides):
        gabor_kern += cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi * i / sides, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    return gabor_kern

'''
Gabor noise
- randomly distributed kernels
- anisotropic when sides = 1, pseudo-isotropic for larger "sides"
'''
def gaborN_rand(size, grid, num_kern, ksize, sigma, theta, lambd, xy_ratio = 1, sides = 1, seed = 0):
    np.random.seed(seed)
    
    # Gabor kernel
    if sides != 1: gabor_kern = gaborK(ksize, sigma, theta, lambd, xy_ratio, sides)
    else: gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    
    # Sparse convolution noise
    sp_conv = np.zeros([size, size])
    dim = int(size / 2 // grid)
    noise = []
    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid + size / 2 - grid / 2
            y = j * grid + size / 2 - grid / 2
            for _ in range(num_kern):
                dx = np.random.randint(0, grid)
                dy = np.random.randint(0, grid)
                while not valid_position(size, x + dx, y + dy):
                    dx = np.random.randint(0, grid)
                    dy = np.random.randint(0, grid)
                weight = np.random.random() * 2 - 1
                sp_conv[int(x + dx)][int(y + dy)] = weight
    
    sp_conv = cv2.filter2D(sp_conv, -1, gabor_kern)
    return normalize(sp_conv)

'''
Gabor noise
- controlled, uniformly distributed kernels

grid        ideally is odd and a factor of size
thetas    orientation of kernels, has length (size / grid)^2
'''
def gaborN_uni(size, grid, ksize, sigma, lambd, xy_ratio, thetas):
    sp_conv = np.zeros([size, size])
    temp_conv = np.zeros([size, size])
    dim = int(size / 2 // grid)
    
    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid + size // 2
            y = j * grid + size // 2
            temp_conv[x][y] = 1
            theta = thetas[(i + dim) * dim * 2 + (j + dim)]
            
            # Gabor kernel
            gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
            sp_conv += cv2.filter2D(temp_conv, -1, gabor_kern)
            temp_conv[x][y] = 0
    
    return normalize(sp_conv)

'''
Perlin noise
- with sine color map
'''
def perlin(size, period, octave, freq_sine, lacunarity = 2):
    
    # Perlin noise
    noise = np.empty((size, size), dtype = np.float32)
    for x in range(size):
        for y in range(size):
            noise[x][y] = pnoise2(x / period, y / period, octaves = octave, lacunarity = lacunarity)
            
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    return normalize(noise)


### Visualize Image ###
'''
Color image

img              has dimension 2 or 3, pixel range [0, 1]
color            is [a, b, c] where a, b, c are from {-1, 0, 1}
'''
def colorize(img, color = [1, 1, 1]):
    if img.ndim == 2: # expand to include color channels
        img = np.expand_dims(img, 2)
    return (img - 0.5) * color + 0.5 # output pixel range [0, 1]

# Plot images in different colors
def plot_colored(img, title):    
    fig = plt.figure(figsize = (20, 6.5))
    plt.subplots_adjust(wspace = 0.05)
    plt.title(title, size = 20)
    plt.axis('off')
    
    ax = fig.add_subplot(1, 4, 1)
    ax.set_title('Black & White', size = 16)
    ax.axis('off')
    plt.imshow(colorize(img, color = [1, 1, 1]))
    
    ax = fig.add_subplot(1, 4, 2)
    ax.set_title('Red & Cyan', size = 16)
    ax.axis('off')
    plt.imshow(colorize(img, color = [1, -1, -1]))
    
    ax = fig.add_subplot(1, 4, 3)
    ax.set_title('Green & Magenta', size = 16)
    ax.axis('off')
    plt.imshow(colorize(img, color = [-1, 1, -1]))
    
    ax = fig.add_subplot(1, 4, 4)
    ax.set_title('Blue & Yellow', size = 16)
    ax.axis('off')
    plt.imshow(colorize(img, color = [-1, -1, 1]))

# Plot power spectrum of image
def plot_spectral(img, title):
    fig = plt.figure(figsize = (20, 6.5))
    plt.subplots_adjust(wspace = 0.05)
    plt.title(title, size = 20)
    plt.axis('off')
    
    # Original image (spatial)
    ax = fig.add_subplot(1, 4, 1)
    ax.set_title('Spatial Domain', size = 16)
    ax.axis('off')
    plt.imshow(img, cmap = plt.cm.gray)
    
    # Original image (spectral)
    ax = fig.add_subplot(1, 4, 2)
    ax.set_title('Power Spectrum', size = 16)
    ax.axis('off')
    plt.imshow(100 * abs(np.fft.fftshift(np.fft.fft2(img))), cmap = plt.cm.gray)
        
    # Original image (spectral variance)
    ax = fig.add_subplot(1, 4, 3)
    ax.set_title('Spectral Variance', size = 16)
    ax.axis('off')
    mean = np.mean(img)
    spec_var = np.fft.fft2(np.square(img -  mean))
    plt.imshow(100 * abs(np.fft.fftshift(spec_var)), cmap = plt.cm.gray)
    
    # Normalized variance
    ax = fig.add_subplot(1, 4, 4)
    ax.set_title('Variance Normalized Image', size = 16)
    ax.axis('off')
    img = normalize_var(img)
    plt.imshow(img, cmap = plt.cm.gray)
   

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

def worley_noise(size, num_points, freq_sine):
    img_grid = create_img_grid(size, size)
    rand_points = select_random_points(size, size, num_points)
    for y in range(size):
        for x in range(size):
            bw_colour = map_to_bw_colour(
                            get_normalized_distance_from_nearest_point(
                                x, y, size, size, rand_points))
            # NOTE: Colour is in RGBA.
            img_grid[y][x] = tuple(([ round(bw_colour) ] * 3) + [ 255 ])
    for x, y in rand_points:
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
    return noise_color



