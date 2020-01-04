# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:15:35 2019

@author: Sandra
"""

# Importacion modulos
import pathlib
import numpy as np
import tensorflow as tf
from functions import process_path, prepare_for_training, timeit


## Carpeta con las imagenes
img_path = 'C:/Users/Sandra/Desktop/TFM2/data/'

# Trabajaremos con las imagenes de train para analizar la performance
data_dir = pathlib.Path(img_path+'TRAIN/')
image_count = len(list(data_dir.glob('*/*.jpg')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

# Definimos un ejemplo de width y height de entrada y dimension del batch
BATCH_SIZE = 64
IMG_HEIGHT = 256
IMG_WIDTH = 256
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
AUTOTUNE = tf.data.experimental.AUTOTUNE
default_timeit_steps = 1000

# Definimos antes BATCH_SIZE, AUTOTUNE, default_timeit_steps, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES



# Utilizando Keras
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode='binary')

print(timeit(ds=train_data_gen, steps=100, BATCH_SIZE=BATCH_SIZE))

# Utilizando tf.data
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
for f in list_ds.take(5):
    print(f.numpy())

labeled_ds = list_ds.map(lambda x: process_path(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", CLASS_NAMES[label.numpy()])

train_ds = prepare_for_training(ds=labeled_ds, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

print(timeit(ds=train_ds, steps=100, BATCH_SIZE=BATCH_SIZE))




