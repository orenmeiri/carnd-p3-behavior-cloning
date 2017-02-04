# import os
# os.environ["DISPLAY"] = ":99"

import random

import cv2
import matplotlib
import numpy as np
import csv
import scipy.ndimage
import scipy.misc
import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from numpy.core.defchararray import lstrip
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from alexnet import AlexNet
import time
from numpy import genfromtxt
from normalize import normalize_image


SIDE_CAMERA_OFFSET = 0.25
EPOCHS = 20
BATCH_SIZE = 192

def preprocess_filename(filename):
    file_name = filename.decode("utf-8")
    file_name = file_name.strip()
    fullname = files_prefix + file_name
    return fullname

files_prefix = "udacity_data/"
lines = genfromtxt(files_prefix + 'driving_log.csv', delimiter=',', dtype="|S50, |S50, |S50, float, float, float, float")
lines = np.array(lines)
source_image_names = []
source_steering_angles = []
for line in lines:
    steering_angle = line[3]
    center_name = preprocess_filename(line[0])
    left_name = preprocess_filename(line[1])
    right_name = preprocess_filename(line[2])
    source_image_names.append(center_name)
    source_steering_angles.append(steering_angle)
    source_image_names.append(left_name)
    source_steering_angles.append(steering_angle + SIDE_CAMERA_OFFSET)
    source_image_names.append(right_name)
    source_steering_angles.append(steering_angle - SIDE_CAMERA_OFFSET)

source_image_names, source_steering_angles = shuffle(source_image_names, source_steering_angles)
source_image_names = np.array(source_image_names)
source_steering_angles = np.array(source_steering_angles)
print("number of training samples: {}".format(source_image_names.shape[0]))


# this function is copied from this student submission https://github.com/ksakmann/CarND-BehavioralCloning/blob/master/model.py
def random_shear(image, steering, shear_range):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    #    print('dx',dx)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering += dsteering
    return image, steering

# def random_roll(image, steering_angle, start_range, end_range):
#     random_roll = np.random.randint(start_range, end_range)
#     image = np.roll(image, random_roll, axis=1)
#     return image, steering_angle + random_roll / 400

def augment_brightness_camera_images(image):
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    img_enhanced = enhancer.enhance(0.5)
    return np.array(img_enhanced)

# saved_image_index = 0

def sample_for_index(index):
    # print("index{}".format(index))
    fullname = source_image_names[index]
    steering_angle = source_steering_angles[index]
    image = scipy.ndimage.imread(fullname)
    random_flip = random.randint(0,1)
    if (random_flip == 1):
        image = scipy.fliplr(image)
        steering_angle = -steering_angle
    return image, steering_angle * 100

def augment_image(image, steering_angle):
    image = augment_brightness_camera_images(image)
    image, steering_angle = random_shear(image, steering_angle, 100)
    return image, steering_angle

def generate_batch(batch_size):
    # global saved_image_index
    while True:
        features = []
        results = []
        weights = []
        for i in range(len(source_image_names)):
            image, steering_angle = sample_for_index(i)
            # should_save = i % 1000
            # if should_save:
            #     saved_image_index = saved_image_index + 1
            #     scipy.misc.imsave("image{}_before.jpg".format(i), image)
            image, steering_angle = augment_image(image, steering_angle)
            # if should_save:
            #     scipy.misc.imsave("image{}_after.jpg".format(i), image)
            image = normalize_image(image)
            features.append(image)
            results.append(steering_angle)
            weights.append(abs(steering_angle + 0.1))

            if (len(features) >= batch_size):
                x = np.array(features)
                y = np.array(results)
                w = np.array(weights)
                features = []
                results = []
                weights = []
                yield x, y, w

from keras.models import Sequential
model = Sequential()
# TODO: Build a Multi-layer feedforward neural network with Keras here.
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(80, 160, 3)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

import gc; gc.collect()

def save_model(filename):
    model.save(filename+".h5")


# generator = generate_image(files_prefix)
generator = generate_batch(BATCH_SIZE)

samples_per_epoch = len(source_image_names)
# samples_per_epoch = 1000

print('samples_per_epoch {}'.format(samples_per_epoch))

model.compile('adam', 'mse')
for i in range(EPOCHS):
    print("REAL epoch: {}/{}".format(i+1, EPOCHS))
    model.fit_generator(generator, samples_per_epoch, 1, verbose=2)
    val_features = []
    val_results = []
    for j in range(int(samples_per_epoch / 100)):
        index = random.randint(0, len(source_image_names)-1)
        val_image,val_steering_angle = sample_for_index(index)
        val_image, val_steering_angle = augment_image(val_image, val_steering_angle)
        val_image = normalize_image(val_image)
        val_features.append(val_image)
        val_results.append(val_steering_angle)
    X_val = np.array(val_features)
    y_val = np.array(val_results)
    # predictions = model.predict(X_val)
    metrics = model.evaluate(X_val, y_val, verbose=2)
    print("validate metrics: {}".format(metrics))
    save_model("model{}".format(i+1))
# history = model.fit(X_train, y_train, nb_epoch=50, validation_split=0.2)

