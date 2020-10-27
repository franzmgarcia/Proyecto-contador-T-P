# Importamos las librerias necesarias, en este caso, para que el codigo funcione correctamente
# esta implementado en 
# Keras 2.3.1
# TensorFlow 1.14
# Python 3.6
# Ademas de las librerias: numpy, pandas, scipy, matplotlib, pillow
# en este caso tiene ademas cv2 pero no es necesario 
# El codigo esta basado en el realizado en el curso de redes neuronales convolucionales de DeepLearning.AI

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
import cv2
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from functions import yolo_filter_boxes, yolo_non_max_suppression, yolo_eval, predict
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
conteo_personas = 0
conteo_carros=0
conteo_buses=0
k=0
num=8

# Iniciamos la sesion
sess = K.get_session()

# Leemos los nombres de las clases de la base de datos coco
class_names = read_classes("model_data/coco_classes.txt")

# Leemos el tamaño de los anchors del modelo yolo que seran un arreglo
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    

# cargamos el modelo del yolo que es un archivo .h5
yolo_model = load_model("model_data/yolo.h5")

# esto nos regresa box_confidence, box_xy, box_wh, box_class_probs
# basicamente con esto podemos hacer la deteccion con las cajas (anchors) que genera el modelo YOLO 
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# Basicamente todo el codigo se basa en:
# Buscamos en la imagen las clases (autobus, carros, etc), para esto segmentamos la imagen y hacemos cajas donde puedan estar introducidos estos
# objetos. Si la probabilidad (score) es superior a 0.6, decimos que la clase encontrada la contamos

# Esta funcion en particular devuelve 3 vectores con: 
# scores de probabilidad para el modelo
# numero de cajas para el modelo
# Las clases que van a estudiarse
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

# Si queremos probar en un video la solucion seria asi:
cap = cv2.VideoCapture('nombrevideo.mp4')

# Mientras este en reproduccion el video
while (cap.isOpened()):

    # leemos el frame

    ret, frame = cap.read()

    # El frame lo escribimos en la carpeta de imagenes para que despues sea leido por el modelo

    cv2.imwrite('images/frame.jpg',frame)

    # no hay problema de que se nos llene de imagenes porque al final eso se va a reescribir

    file_name="frame.jpg"

    # aca tenemos los resultados de la funcion predict, que es la que finalmente nos da los resultados en funcion de la imagen

    out_scores, out_boxes, out_classes = predict(sess,file_name,class_names,yolo_model,scores, boxes, classes)

    # vemos las probabilidades
    print(out_scores)

    # aca hacemos el contador
    # vemos que el label para personas es 0
    # carros: 2 o 67
    # buses: 5
    # para revisar los demas, solo hay que revisar la lista de clases suministradas

    if len(out_classes)>0:

        for i in range(len(out_classes)):
            if out_classes[i]==0:
                conteo_personas +=1
            if out_classes[i]==2 or out_classes[i]==67:
                conteo_carros +=1
            if out_classes[i]==5:
                conteo_buses +=1
    
    # Finalmente aca determinamos cuanto de cada uno de estas clases habia, finalmente hacemos 0 todo de nuevo
    print("Personas: "+str(conteo_personas))
    print("Carros: "+str(conteo_carros))
    print("Buses: "+str(conteo_buses))

    conteo_personas = 0
    conteo_carros=0
    conteo_buses=0
