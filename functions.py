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
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    
    """

    Tenemos 2 variables para determinar si hay una clase dentro de una de las cajas
    La primera box_confidence, indica la probabilidad de haber encontrado una clase dentro de una de las cajas
    La segunda box_class_probs, indica la probabilidad de que dentro de esa caja este la clase que buscamos

    """

    # Determinamos el producto de ambos, que va a ser la probabilidad de que dentro de una caja este la clase
    
    box_scores = (box_confidence * box_class_probs)
    


    # Encontramos de todas las clases seleccionadas la que tiene una mayor probabilidad, para decir que en esa caja esta esa clase
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1,  keepdims=False)
    

    # aca filtramos de acuerdo al valor del threshold, arriba el threshold esta definido como 0.6
    # Si el valor obtenido del producto de box_class_scores, que es el maximo, no es mayor a 0.6, no validamos la existencia de una clase
    # dentro de la caja
    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    # Este valor de threshold es quizas lo que se deberia revisar mas a la hora de perfeccionar algo en el codigo
    # Threshold va de 0 a 1

    return scores, boxes, classes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    """

    Aca usamos el metodo de non-maximum suppression, que no es mas que el metodo de interseccion sobre la union
    Que nos va a permitir definir un nuevo threshold para confirmar si efectivamente tenemos una clase en imagen

    """

    # Definimos e inicializamos el tensor
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 
    
    # Seleccionamos los indices que superan la non-max suppression
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold,name=None)

    # Seleccionamos unicamente los indices de las cajas que superan la non_max suppression
    
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Aca aplicamos todas las funciones antes definidas, para hacer el filtrado por threshold y para hacer 
    el non-max suppression

    """
    
    # Obtenemos las probabilidades y los tamaños de las cajas
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convertimos las cajas para que sean los bordes de las cajas

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Filtramos a traves del threshold antes definido
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, iou_threshold)
    
    # escalamos a la imagen original
    boxes = scale_boxes(boxes, image_shape)

    # Aplicamos non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    
    return scores, boxes, classes

def predict(sess, image_file,class_names,yolo_model,scores, boxes, classes):
    
    """
    Aca hacemos el procesamiento de la imagen
    El valor del reshape es 608 x 608 porque con imagenes de ese tamaño fue entrenada la red

    """

    # Seleccionamos la imagen a procesar y hacemos el reshape a 608,608

    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    
    # Iniciamos la sesion de tensorflow
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    

    # Print de las predicciones realizadas
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Colores para determinar las clases 
    colors = generate_colors(class_names)
    # Dibujamos las cajas en la imagen
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    # Podriamos guardar la imagen asi
    #image.save(os.path.join("out", image_file), quality=100)
    
    # Y mostrar los resultados asi
    #output_image = scipy.misc.imread(os.path.join("out", image_file))
    #imshow(output_image)
    
    return out_scores, out_boxes, out_classes