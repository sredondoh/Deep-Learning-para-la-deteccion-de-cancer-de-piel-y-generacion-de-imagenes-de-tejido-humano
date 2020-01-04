# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:27:10 2019

@author: Sandra
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import random
import shutil
import os
import sys
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# functions
def partition_img(m_img, s_train, s_test, s_validation):
    # Subset train data
    train = random.sample(range(len(m_img)),s_train)
    m_train = m_img[train]
    m_img = np.delete(m_img, train)
    # Subset test data
    test = random.sample(range(len(m_img)),s_test)
    m_test = m_img[test]
    m_img = np.delete(m_img, test)
    #Subset validation data
    vald = random.sample(range(len(m_img)),s_validation)
    m_validation = m_img[vald]
    return m_train, m_test, m_validation


def mov_img(path_dest, array, type='TRAIN'):
    # Move all the images from a folder to another 
    for i in range(len(array)):
        path = array[i].split('\\')[-2]
        name = path_dest+type+'\\'+path
        if not os.path.exists(name):
            os.makedirs(name)
        source_file = array[i]
        dest_file = name+'\\'+array[i].split('\\')[-1]
        shutil.copyfile(source_file, dest_file)
        prints='It: '+str(i+1)+' of '+str(len(array))
        sys.stdout.write('\r'+prints)


def get_label(file_path, CLASS_NAMES):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '\\')
    # The second to last is the class-directory
    return int(parts[-2] == CLASS_NAMES[1])


def decode_img(img, IMG_WIDTH, IMG_HEIGHT):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], 
                         method='bilinear', 
                         preserve_aspect_ratio=False,
                         antialias=False,name=None)
    
    
def process_path(file_path, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT):
    label = get_label(file_path, CLASS_NAMES)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, IMG_WIDTH, IMG_HEIGHT)
    return img, label


def prepare_for_training(ds, cache, shuffle_buffer_size, BATCH_SIZE, AUTOTUNE):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


def timeit(ds, steps, BATCH_SIZE):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))


def get_labels(ds,steps, model):
    Y_lab = []
    Y_pred = []
    for i in range(int(steps)):
        img, label = next(iter(ds))
        label = label.numpy()
        label = list(label)
        label_pred = model.predict_proba(img)
        label_pred = label_pred.reshape(1,64)[0]
        label_pred = list(label_pred)
        Y_lab = Y_lab+label
        Y_pred = Y_pred+label_pred
    return Y_lab, Y_pred


def label_y(x):
    if x < 0.5:
        return 0
    else:
        return 1
    
def model_cnn(model, train_ds, STEPS_PER_EPOCH, test_ds, STEPS_PER_EPOCH_TEST, model_name, model_path, EPOCHS):
    # set seed
    tf.random.set_seed(1654)

    # model compile
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    

    #checkpoints
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'best_weights_'+model_name+'.hdf5', 
                                                      monitor = 'val_accuracy',
                                                      verbose=0,
                                                      save_best_only=True)
    # train the model
    history = model.fit(train_ds,
                    steps_per_epoch = STEPS_PER_EPOCH,
                    epochs = EPOCHS,
                    callbacks=[checkpointer],
                    validation_data = test_ds,
                    validation_steps = STEPS_PER_EPOCH_TEST)

    # Save the results
    with open(model_path+'/history_'+model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.load_weights(model_path+'best_weights_'+model_name+'.hdf5')
    model.save(model_path+model_name+'.hdf5')

def validation(model_name, model_folder, STEPS_PER_EPOCH_TEST, test_ds):
    
    # Cargamos el modelo 1: Versión 1
    model = tf.keras.models.load_model(model_folder+model_name+'.hdf5')
    print(model.summary())
    with open(model_folder+'history_'+model_name, 'rb') as f:
        history_1 = pickle.load(f)
        
    # Definimos el historico de accuracy y de loss
    acc = history_1['accuracy'].copy()
    val_acc = history_1['val_accuracy'].copy()
    loss = history_1['loss'].copy()
    val_loss = history_1['val_loss'].copy()
    
    print('Best epoch:'+str(np.argmax(val_acc)))
    print('Best val_acc:'+str(max(val_acc)))

    # Hacemos el grafico
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(model_folder+model_name+'.png')
    
    # Validación de los resultados
    for i in range(int(STEPS_PER_EPOCH_TEST)):
        img, label = next(iter(test_ds))
    
    Y_lab, Y_pred = get_labels(test_ds, STEPS_PER_EPOCH_TEST, model)
    Y_p_label = [label_y(elm) for elm in Y_pred]
    
    print('Confusion Matrix')
    print(confusion_matrix(Y_lab, Y_p_label))
    print('Classification Report')
    target_names = ['Benign', 'Malignant']
    print(classification_report(Y_lab, Y_p_label, target_names=target_names))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(Y_lab, Y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(model_folder+'roc_'+model_name+'.png')


####### Data augmentation
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.dtypes.int32))

def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(256, 256))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.dtypes.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.dtypes.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def zoom2(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(128, 128))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.dtypes.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.dtypes.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def process_path_Aug(file_path, CLASS_NAMES, IMG_WIDTH, IMG_HEIGHT, augmentations = [flip, color, zoom, rotate]):
    label = get_label(file_path, CLASS_NAMES)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, IMG_WIDTH, IMG_HEIGHT)
    for f in augmentations:
        if random.uniform(0,1) > 0.75:
            img = f(img)
    img = tf.clip_by_value(img, 0, 1)
    return img, label

def visualize_filter2(model, layer_name):
    for layer in model.layers:
    # check for convolutional layer
        if layer_name!=layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
  
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_chanels = filters.shape[2]
    n_filters = filters.shape[3]
    images_per_row = n_filters
    size = filters.shape[1]
    n_cols = n_chanels
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for row in range(images_per_row):
        f = filters[:, :, :, row]
        for col in range(n_cols):
            ff = f[:, :, col]
            display_grid[col * size : (col + 1) * size, # Displays the grid
                      row * size : (row + 1) * size] = ff
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                  scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.axes(yticklabels=[], xticklabels=[])
    for row in range(images_per_row):
        plt.axvline((row*size)+2.5)
    for col in range(n_cols):
        plt.axhline((col*size)+2.5)
    plt.grid(False)
    plt.ylabel('chanels')
    plt.xlabel('filters')
    plt.imshow(display_grid,aspect='auto', cmap='gray')
    
def process_path_gan(file_path, IMG_WIDTH, IMG_HEIGHT):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img =  tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT], 
                         method='bilinear', 
                         preserve_aspect_ratio=False,
                         antialias=False,name=None)
    return img

def prepare_for_training_gan(ds, cache, BATCH_SIZE, shuffle_buffer_size, AUTOTUNE):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds