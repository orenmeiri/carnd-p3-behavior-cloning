import matplotlib
import numpy as np
import scipy
import cv2

def rgb2gray(rgb):
    return np.inner(rgb[...,:3], [[0.299, 0.587, 0.114]])

def channelSplit(image):
    return np.dsplit(image,image.shape[-1])

def convert_hsv(rgb_image):
    hsvImage = matplotlib.colors.rgb_to_hsv(rgb_image)
    [h, s, v]=channelSplit(hsvImage)
    h = np.reshape(h, (160, 320))
    s = np.reshape(s, (160, 320))
    v = np.reshape(v, (160, 320))
    return h, s, v


def normalize_image(image):
    image = scipy.misc.imresize(image, (80, 160))
    return image
