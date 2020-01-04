# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:31:21 2019

@author: Sandra
"""

# Importación modulos
import os
import numpy as np
import glob
import pathlib
import json
import shutil
import sys
import random
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


# Importación funciones
from functions import partition_img, mov_img

# Carpeta de destino
folder_dest = 'C:/Users/Sandra/Desktop/TFM2/'
folder_data = folder_dest+'data/'
folder_json = folder_dest+'ISIC-json/'

# Carpeta con las imagenes del ISIC archive
path_img = 'C:/Users/Sandra/Desktop/TFM2/ISIC-images/ISIC-images/'

# Si la carpeta no existe la creamos
if not os.path.exists(folder_data):
    os.makedirs(folder_data)
    print('Creating folder: '+folder_data)

# Si aun no hemos movido las imagenes a la carpeta de destino:
if not len(np.array(glob.glob(folder_data+'*/*/*.jpg')))==4394:
  
    # 1.Leer las imagenes del ISIC-archive
    
    # Filtrarlas por dermatoscopio y clasificarlas por benignas, malignas o otros
    path = pathlib.Path(path_img+'*/*.jpg') #img en ISIC .jpg
    list_img = np.array(glob.glob(str(path)))
    
    # Creamos la carpeta json
    if not os.path.exists(folder_json):
        os.makedirs(folder_json)
    
    i=0
    for path in list_img:
        img_name = path.split('\\')[-1]
        
        # Type: read json file
        if os.path.isfile(path.replace('.jpg','.json')):
            path_json = path.replace('.jpg','.json')
            with open(path_json) as json_file:
                data = json.load(json_file)
            type_img = data.get('meta').get('clinical').get('benign_malignant')
            is_clinical = data.get('meta').get('acquisition').get('image_type')
            
            # Copy the json file
            source_file_json = path.replace('.jpg','.json')
            dest_file_json = path_json+img_name.replace('.jpg','.json')
            shutil.copyfile(source_file_json, dest_file_json)
        
        # Move the .jpg file to new folder
        if not is_clinical == 'clinical':
            if type_img=='malignant' or  type_img=='benign':
                
                # Creamos una carpeta dentro de ISIC-img con malignant, benign y other
                name_folder = folder_dest+'ISIC-img/'+type_img
                
                # Si no existe lo creamos
                if not os.path.exists(name_folder):
                    os.makedirs(name_folder)
                
                # Copiamos la imagen
                source_file = path
                dest_file = name_folder+'/'+img_name
                shutil.copyfile(source_file, dest_file)   
            
            else: 
                name_folder = folder_dest+'ISIC-img/other'
                
                # Create if not exists
                if not os.path.exists(name_folder):
                    os.makedirs(name_folder)
                source_file = path
                dest_file = name_folder+'/'+img_name
                shutil.copyfile(source_file, dest_file)
        
        i=i+1
        prints='Reading: '+str(i)+' of '+str(len(list_img))
        sys.stdout.write('\r'+prints)  

    # 2. Hacer la muestra de los datos: Img malignas todas, benignas submuestra misma dimension
    random.seed(164754)
    malignant_path = folder_dest.replace('/','\\')+'ISIC-img\\malignant\\*'
    m_img = np.array(glob.glob(malignant_path))

    # Dimension
    s_train = int(0.7*len(m_img))
    s_test = int(0.2*len(m_img))
    s_validation = len(m_img) - s_train - s_test

    # For malignant
    m_train, m_test, m_validation = partition_img (m_img, s_train, s_test, s_validation)

    # For benign
    benign_path = folder_dest.replace('/','\\')+'ISIC-img\\benign\\*'
    b_img = np.array(glob.glob(benign_path))
    
    b_train, b_test, b_validation = partition_img (b_img, s_train, s_test, s_validation)

    # Finally the data
    train = np.append(m_train,b_train); #del m_train,b_train
    test = np.append(m_test,b_test); #del m_test,b_test
    validation = np.append(m_validation,b_validation); #del m_validation,b_validation

    # shuffle
    train = train[random.sample(range(len(train)),len(train))]
    test = test[random.sample(range(len(test)),len(test))]
    validation = validation[random.sample(range(len(validation)),len(validation))]

    # 3. Movemos las imagenes selecionadas a la carpeta data de TFM
    mov_img(path_dest=folder_data.replace('/','\\'),array=validation, type='VALIDATION')
    mov_img(path_dest=folder_data.replace('/','\\'),array=train, type='TRAIN')
    mov_img(path_dest=folder_data.replace('/','\\'),array=test, type='TEST')


# Dimension
s_train = len(np.array(glob.glob(folder_data.replace('/','\\')+'TRAIN\\*\\*.jpg')))
s_test = len(np.array(glob.glob(folder_data.replace('/','\\')+'TEST\\*\\*.jpg')))
s_validation = len(np.array(glob.glob(folder_data.replace('/','\\')+'VALIDATION\\*\\*.jpg')))

print('Img train: '+str(s_train)+', Img test: '+str(s_test)+', Img validation: '+str(s_validation)+': Total: '+str(s_train+s_test+s_validation))

# Analizamos la dimension de las imagenes 
if not os.path.isfile(folder_dest+'dist_img.csv'):
    W = []
    H = []
    i=0
    # Summary de las imagenes:
    # All the json files
    path = folder_data.replace('/','\\')+'*\\*\\*.jpg'
    list_img = np.array(glob.glob(path))
    for file in list_img:
        img = Image.open(file)
        width, height = img.size
        W.append(width)
        H.append(height)
        prints='It:'+str(i+1)+' of:'+str(len(list_img))
        sys.stdout.write('\r'+prints)
        i=i+1

    #create the dataframe
    x = pd.DataFrame(list(zip(W,H)),columns=['width_v','height_v'])
    x_freq = pd.crosstab(x.width_v,x.height_v).replace(0,np.nan).      stack().reset_index().rename(columns={0:'Count'})
    x_freq = x_freq.sort_values(by=['Count'],ascending=False)
    x_freq.to_csv(folder_dest+'dist_img.csv',sep=',',index=False)

x_freq = pd.read_csv(folder_dest+'dist_img.csv')
plt.scatter(x_freq.width_v, x_freq.height_v, c=x_freq.Count, cmap='jet', alpha=0.5)
plt.colorbar()  # show color scale
plt.savefig(folder_dest+'dist_w_h.png')