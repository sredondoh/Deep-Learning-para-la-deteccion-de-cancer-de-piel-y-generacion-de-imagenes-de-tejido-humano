# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:08:58 2019

@author: Sandra
"""
# https://www.tensorflow.org/tutorials/generative/dcgan

# ### 1.1 Configuración del entorno y liberías
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print(tf.__version__)
from PIL import Image
import numpy as np
import glob
import random
import os
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import PIL
import time
from IPython import display
from functions import process_path_gan, flip, color, zoom2, rotate, prepare_for_training_gan


# Path con las imagenes
folder = 'C:/Users/Sandra/Desktop/TFM2/'

# Definimos los parametros
BATCH_SIZE = 64
IMG_HEIGHT = tf.constant(128, dtype=tf.int64)
IMG_WIDTH = tf.constant(128, dtype=tf.int64)
AUTOTUNE = tf.data.experimental.AUTOTUNE
noise_dim = 100
shuffle_bufferS = len(np.array(glob.glob(folder+'ISIC-images/ISIC-images/*/*.jpg')))
print(shuffle_bufferS)

# Importamos las imagenes
gan_ds = tf.data.Dataset.list_files(folder+'ISIC-images/ISIC-images/*/*.jpg').map(lambda x: process_path_gan(x, IMG_WIDTH, IMG_HEIGHT), num_parallel_calls=AUTOTUNE)

# Add augmentations
dataset = gan_ds
augmentations = [flip, color, zoom2, rotate]

for f in augmentations:
    dataset = dataset.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x), num_parallel_calls=4)
dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1))

gan_dataset = prepare_for_training_gan(ds=dataset, cache=True, BATCH_SIZE=BATCH_SIZE, shuffle_buffer_size=shuffle_bufferS, AUTOTUNE=AUTOTUNE)

# Definimos el modelo generativo
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(32*32*32, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((32, 32, 32)))
    assert model.output_shape == (None, 32, 32, 32) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, IMG_WIDTH, IMG_HEIGHT, 3) 

    return model

generator = make_generator_model()
noise = tf.random.normal([1, noise_dim])
generated_image = generator(noise, training=False)
plt.imshow(generated_image.numpy()[0], cmap='gray', interpolation='nearest')

generator.summary()


# Definimos el discriminador
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IMG_WIDTH.numpy(), IMG_HEIGHT.numpy(), 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
discriminator.summary()

# Definimos la funcion de perdida del generador y discirminador
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# El optimizador
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#Los checkpoints
checkpoint_dir = 'C:/Users/Sandra/Desktop/TFM2/models/training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Ejecucion del modelo
EPOCHS = 50
num_examples_to_generate = 4
# Semilla para seguir como va generando las imagenes
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i+1)
        #plt.imshow(predictions[i, :, :, :], interpolation='nearest')
        #plt.imshow(predictions[i], interpolation='nearest')
        plt.imshow(predictions[i])
            
        plt.axis('off')
  
    if not os.path.exists('C:/Users/Sandra/Desktop/TFM2/models/image_at_epoch'):
        os.makedirs('C:/Users/Sandra/Desktop/TFM2/models/image_at_epoch')
  
    display.clear_output(wait=True)
    plt.savefig('C:/Users/Sandra/Desktop/TFM2/models/image_at_epoch/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

train(gan_dataset, 1000)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#'C:/Users/Sandra/Desktop/TFM2/models/training_checkpoints/ckpt-66'

# Generacion del Gif y mostramos la ultima imagen generada
# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('C:/Users/Sandra/Desktop/TFM2/models/image_at_epoch/image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(880)

anim_file = 'C:/Users/Sandra/Desktop/TFM2/models/dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('C:/Users/Sandra/Desktop/TFM2/models/image_at_epoch/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

    import IPython
    if IPython.version_info > (6,2,0,''):
        display.Image(filename=anim_file)

# Guardamos los resultados
generator.save('C:/Users/Sandra/Desktop/TFM2/models/generator.h5')
discriminator.save('C:/Users/Sandra/Desktop/TFM2/models/discriminator.h5')

# # Evaluación de los resultados de la GAN
generator = tf.keras.models.load_model('C:/Users/Sandra/Desktop/TFM2/models/generator.h5')

# Generar y guardar 100 imagenes
import matplotlib

if not os.path.exists('C:/Users/Sandra/Desktop/TFM2/data_GAN'):
    os.makedirs('C:/Users/Sandra/Desktop/TFM2/data_GAN')

for i in range(100):
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    path = "C:/Users/Sandra/Desktop/TFM2/data_GAN/"+str(i)+".jpg"
    matplotlib.image.imsave(path, generated_image.numpy()[0])

# Selecionar al azar 100 imagenes reales
if not os.path.exists('C:/Users/Sandra/Desktop/TFM2/data_Normal128'):
    os.makedirs('C:/Users/Sandra/Desktop/TFM2/data_Normal128')

img = np.array(glob.glob('C:/Users/Sandra/Desktop/TFM2/ISIC-images/ISIC-images/*/*.jpg'))

#sample 100
img = img[random.sample(range(len(img)),100)]
i=1
for im_file in img:
    im = Image.open(im_file)
    im = im.resize((128,128), Image.BILINEAR)
    path = "C:/Users/Sandra/Desktop/TFM2/data_Normal128/"+str(i)+".jpg"
    i = i+1
    im.save(path)

# Funcion que extrae la imagen y el tipo
def read_img_GAN(file_path):
    type_img = file_path.split('\\')[-2]
    if type_img=='data_Normal128':
        type_img = 'Real'
    else:
        type_img = 'Generada' 
    img = Image.open(file_path)
    img1 = np.asarray(img)
    img1 = img1.flatten()
    return type_img, img1

img_n = list(glob.glob('C:\\Users\\Sandra\\Desktop\\TFM2\\data_Normal128\\*.jpg'))
img_g = list(glob.glob('C:\\Users\\Sandra\\Desktop\\TFM2\\data_GAN\\*.jpg'))
img = img_n + img_g
types = []
img_ = []
for files in img:
    type_img, img1 = read_img_GAN(files)
    types.append(type_img)
    img_.append([img1])

img_ = np.array(img_)
img_ = np.reshape(img_, (200, 49152))
df = pd.DataFrame(img_)

# Computo del PCA sobre el df anterior
from sklearn import preprocessing
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df['target'] = types

features = df.columns[0:-1]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)
pca.explained_variance_ratio_
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

# Plot de los resultados
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Componente principal 1', fontsize = 15)
ax.set_ylabel('Componente principal 2', fontsize = 15)
ax.set_title('PCA de imágenes 128x128px a color', fontsize = 20)
targets = ['Real', 'Generada']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 0]
               , finalDf.loc[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

