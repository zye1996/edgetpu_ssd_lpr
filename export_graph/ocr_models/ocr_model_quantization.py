import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import *
from keras.layers import *

def model_seq_rec_cnn(model_path):
    input_tensor = Input((160, 40, 3))
    x = input_tensor
    for i in range(3):
        x = Conv2D(32*(2**i), 3, strides=1, padding='SAME')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
    x = Conv2D(256, 5, strides=1, padding='VALID')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, 1, strides=1, padding='VALID')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(84, 1, strides=1, padding='VALID')(x)
    x = Reshape((16, 84))(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)
    return base_model

def gen():
    dir = "../../images_rec/ocr_images"
    for img_name in os.listdir(dir):
        img = plt.imread(os.path.join(dir, img_name))
        img = cv2.resize(img,(160, 40))
        img = img.transpose(1, 0, 2)
        yield [img[np.newaxis, :, :, :].astype(np.float32)]

# load weights
model = model_seq_rec_cnn("ocr_plate_all_w_rnn_2.h5")
model.save("ocr_model.h5")

# quantization
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("ocr_model.h5")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = gen
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()
open('ocr_model_quant_int8.tflite', 'wb').write(tflite_quant_model)