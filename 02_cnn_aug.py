# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:07:38 2019

@author: Sandra
"""
import pathlib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from functions import prepare_for_training, validation, model_cnn, process_path_Aug

# Path con las imagenes
folder = 'C:/Users/Sandra/Desktop/TFM2/'
path = pathlib.Path('C:/Users/Sandra/Desktop/TFM2/data/')

# Definimos los parametros
BATCH_SIZE = 64
IMG_HEIGHT = tf.constant(256, dtype=tf.int64)
IMG_WIDTH = tf.constant(256, dtype=tf.int64)
CLASS_NAMES = np.array([item.name for item in path.glob('TRAIN/*') if item.name != "LICENSE.txt"])
CLASS_NAMES = tf.constant(CLASS_NAMES)
AUTOTUNE = tf.data.experimental.AUTOTUNE

#################################################### DATA AUGMENTATION

#import matplotlib.pyplot as plt
#plt.figure(figsize=(10,10))
#x = labeled_ds.take(25)
#i = 0
#for img, label in x:
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(img.numpy(), cmap=plt.cm.binary)
#    # The CIFAR labels happen to be arrays, 
#    # which is why you need the extra index
#    if label.numpy() == 1:
#        label = CLASS_NAMES.numpy()[1]
#    else:
#        label = CLASS_NAMES.numpy()[0]
#    plt.xlabel(label.decode('UTF-8'))
#    i = i +1 
#plt.show()

# Importamos TRAIN
data_dir = pathlib.Path(str(path)+'/TRAIN/')
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_ds = list_ds.map(lambda x: process_path_Aug(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)
train_ds = prepare_for_training(ds=labeled_ds, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

# Importamos Validación 
data_dir = pathlib.Path(str(path)+'/TEST/')
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH_TEST = np.ceil(image_count/BATCH_SIZE)

list_ds_test = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_ds_test = list_ds_test.map(lambda x: process_path_Aug(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)
test_ds = prepare_for_training(ds=labeled_ds_test, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

# Importamos Test
data_dir = pathlib.Path(str(path)+'/VALIDATION/')
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH_VAL = np.ceil(image_count/BATCH_SIZE)

list_ds_val = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_ds_val = list_ds_test.map(lambda x: process_path_Aug(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)
val_ds = prepare_for_training(ds=labeled_ds_test, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

# Creamos una carpeta con los modelos
if not os.path.exists(folder+'models'):
    os.makedirs(folder+'models')

## Verison 1 ajustada con regularizadores l1   
if not os.path.isfile(folder+'models/model_4.hdf5'):
    tf.random.set_seed(1647)
    # Define the model
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT.numpy(), IMG_WIDTH.numpy(),3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Summary the model
    model.summary()
    
    # Generate the model
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_4'
    model_cnn(model, train_ds, STEPS_PER_EPOCH, test_ds, STEPS_PER_EPOCH_TEST, model_name, model_folder, EPOCHS=100)
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)    

else:
    # Print the validation
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_4'
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)
