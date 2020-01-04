# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:44:22 2019

@author: Sandra
"""
# Importacion modulos
import tensorflow as tf
import pathlib
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from functions import visualize_filter2
import numpy as np
from keras.preprocessing import image

path = pathlib.Path('C:/Users/Sandra/Desktop/TFM2/data/')
CLASS_NAMES = np.array([item.name for item in path.glob('TRAIN/*') if item.name != "LICENSE.txt"])

model = tf.keras.models.load_model('C:/Users/Sandra/Desktop/TFM2/models/model_3.hdf5')
model.summary()


# #### 4.3.1 Filtros del modelo
visualize_filter2(model, layer_name='conv2d')
visualize_filter2(model, layer_name='conv2d_1')
visualize_filter2(model, layer_name='conv2d_2')

# #### 4.3.2 Intermediate activations
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
img_path = 'C:/Users/Sandra/Desktop/TFM2/data/TRAIN/malignant/ISIC_0000002.jpg'
img = image.load_img(img_path, target_size=(256, 256))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)

# predicting images
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class is:",CLASS_NAMES[classes[0]])

layer_outputs = [layer.output for layer in model.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

activations = activation_model.predict(img_tensor) 
# Returns a list of five Numpy arrays: one array per layer activation

activations = activation_model.predict(img_tensor) 
# Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 8], cmap='viridis')


layer_names = []
for layer in model.layers[:12]:
    if ('flatten' not in layer.name) and ('dense' not in layer.name):
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations): 
    # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.axis('off')
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig("feature_map_"+layer_name+".jpg", bbox_inches='tight')