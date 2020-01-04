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
from functions import process_path, prepare_for_training, validation, model_cnn

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

# Importamos TRAIN
data_dir = pathlib.Path(str(path)+'/TRAIN/')
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_ds = list_ds.map(lambda x: process_path(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)
train_ds = prepare_for_training(ds=labeled_ds, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

# Importamos Validación 
data_dir = pathlib.Path(str(path)+'/TEST/')
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH_TEST = np.ceil(image_count/BATCH_SIZE)

list_ds_test = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_ds_test = list_ds_test.map(lambda x: process_path(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)
test_ds = prepare_for_training(ds=labeled_ds_test, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

# Importamos Test
data_dir = pathlib.Path(str(path)+'/VALIDATION/')
image_count = len(list(data_dir.glob('*/*.jpg')))
STEPS_PER_EPOCH_VAL = np.ceil(image_count/BATCH_SIZE)

list_ds_val = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
labeled_ds_val = list_ds_test.map(lambda x: process_path(x, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)
val_ds = prepare_for_training(ds=labeled_ds_test, cache=True, shuffle_buffer_size=1000, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)



# Creamos una carpeta con los modelos
if not os.path.exists(folder+'models'):
    os.makedirs(folder+'models')
    
### MODELO 1
if not os.path.isfile(folder+'models/model_0.hdf5'):
    
    tf.random.set_seed(1647)

    # Model definition 5s x step
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT.numpy(), IMG_WIDTH.numpy() ,3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # summary model
    model.summary()
    
    # Generate the model
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_0'
    model_cnn(model, train_ds, STEPS_PER_EPOCH, test_ds, STEPS_PER_EPOCH_TEST, model_name, model_folder, EPOCHS=100)
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)    

else:
    # Print the validation
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_0'
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)  
    

## Verison 1 ajustada...    
if not os.path.isfile(folder+'models/model_1.hdf5'):
    
    tf.random.set_seed(1647)
    # Model definition 5s x step
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT.numpy(), IMG_WIDTH.numpy(),3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    # summary model
    model.summary()
    
    # Generate the model
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_1'
    model_cnn(model, train_ds, STEPS_PER_EPOCH, test_ds, STEPS_PER_EPOCH_TEST, model_name, model_folder, EPOCHS=100)
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)    

else:
    # Print the validation
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_1'
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)
    

## Verison 1 ajustada con regularizadores l1   
if not os.path.isfile(folder+'models/model_2.hdf5'):
    tf.random.set_seed(1647)
    # Define the model
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT.numpy(), IMG_WIDTH.numpy(),3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Summary the model
    model.summary()
    
    # Generate the model
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_2'
    model_cnn(model, train_ds, STEPS_PER_EPOCH, test_ds, STEPS_PER_EPOCH_TEST, model_name, model_folder, EPOCHS=100)
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)    

else:
    # Print the validation
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_2'
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)

## Verison 1 ajustada con regularizadores l1   
if not os.path.isfile(folder+'models/model_3.hdf5'):
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
    model_name = 'model_3'
    model_cnn(model, train_ds, STEPS_PER_EPOCH, test_ds, STEPS_PER_EPOCH_TEST, model_name, model_folder, EPOCHS=100)
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)    

else:
    # Print the validation
    model_folder = 'C:/Users/Sandra/Desktop/TFM2/models/'
    model_name = 'model_3'
    validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds)
    
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
    
model_name = 'model_3'
model = tf.keras.models.load_model(model_folder+model_name+'.hdf5')
validation(model_name, model_folder, STEPS_PER_EPOCH_VAL, val_ds)
