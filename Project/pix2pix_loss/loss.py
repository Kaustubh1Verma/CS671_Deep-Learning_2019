from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Flatten, Dense, Dropout
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.optimizers import SGD
import cv2
from tensorflow.python.keras.applications import VGG19
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
import numpy as np
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.applications.vgg19 import preprocess_input


model=VGG19(weights="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",include_top=False)
model_extractfeatures = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)

def feature_extract(x):
    fc2_features = model_extractfeatures.predict(x)
    return fc2_features

def preprocess(img):
    print(img.shape)
    cv2.resize(img,(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def smooth_L1_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
   x   = k.abs(y_true - y_pred)
   x   = k.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  k.sum(x)


#def total_loss(y_true, y_pred):
    # img1=image.load_img(y_true_path, target_size=(224, 224))
    # img2=image.load_img(y_pred_path, target_size=(224, 224))
    #f1=preprocess(y_true)
    #f2=preprocess(y_pred)
    #fx1=feature_extract(f1)
    #fx2=feature_extract(f2)
    #loss1 = tf.reduce_mean(tf.squared_difference(fx1, fx2))
    loss2=smoothL1(y_true,y_pred)
    return k.eval(loss2)+k.eval(loss2)
