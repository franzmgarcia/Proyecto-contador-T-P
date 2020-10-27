# Proyecto-Contador-T-P

El proyecto de Contador T-P funciona a través del algoritmo YOLO. El algoritmo YOLO consiste en realizar la segmentación de la imagen a procesar en un número de partes. En cada uno de los segmentos se realiza el proceso de la red neuronal convolucional a fin de conseguir algunas de las clases objetivo en dicho segmento. Para esto además, se definen unas cajas, que son utilizadas para identificar las clases dentro de dichas cajas. Finalmente la red neuronal identifica la posibilidad de que tengas una clase dentro de una de esas cajas (anchor_boxes) y si esta probabilidad es mayor que un umbral, se toma como correcta la existencia de ese elemento dentro de la imagen. Este proceso se realiza con todas las imágenes del vídeo, y se extraen los elementos de interés (En este caso peatones, autobuses, motos, etc)

Estas clases provienen del dataset COCO, y si queremos hacer un conteo lo que debemos realizar es: 
- Identificar las clases
- Sumar el número de veces que aparecen
- Reiniciar el contador para cuando se analice la próxima imagen

También es importante agregar, que para el correcto funcionamiento del proyecto se debe utilizar:

- Tensorflow 1.14.0
- Keras 2.3.0
- Python 3.6

Si no se utilizan estas librerías puede ocurrir que el código no funcione, por errores de compatibilidad entre las librerías, otras librerías importantes son:

- Pandas
- Numpy
- Scilab
- h5py



