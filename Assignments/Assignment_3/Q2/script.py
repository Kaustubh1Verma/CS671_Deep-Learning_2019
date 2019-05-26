import os
import glob
import cv2
import argparse
import argparse
import pickle
import itertools
import numpy as np
from os import listdir
from scipy.io import loadmat
import matplotlib.image as img
from os.path import isfile, join

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D
from tensorflow.python.keras.layers.convolutional import Deconvolution2D
from tensorflow.python.keras.layers import Input, Add, Dropout, Permute, add

from tensorflow.python.keras.models import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import optimizers

import tensorflow as tf
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Lambda, Activation


# Current python dir path

dir_path = os.path.dirname(os.path.realpath('__file__'))
parser = argparse.ArgumentParser()
parser.add_argument("-T", "--test_folder", required=True, help="Folder for the test images")
parser.add_argument("-O", "--output_folder", required=True, help="Folder for the predicted images")

####### read test images path ####################
args = parser.parse_args()
test_folder = os.path.join(dir_path,args.test_folder)
output_folder = os.path.join(dir_path,args.output_folder)

test_images = glob.glob(test_folder+"/"+"*.tiff")
print(test_images)


############### Global variables ########################

model_input_height=224
model_input_width=224

output_height = 300
output_width = 400

nClasses = 2

VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

######################## Create target Directory if don't exist ###########################3
if not os.path.exists(output_folder):
	os.mkdir(output_folder)
	print("Directory " , output_folder ,  " Created ")
else:    
	print("Directory " , output_folder ,  " already exists")


############################## Model #########################################

def FCN(num_classes ,  input_height=224, input_width=224,vgg_weight_path=None):

    img_input = Input(shape=(input_height,input_width, 3))


    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_3_out = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(block_3_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_4_out = MaxPooling2D()(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(block_4_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)
	
    # Load pretrained weights.
    #if vgg_weight_path is not None:
         #vgg16 = Model(img_input, x)
         #vgg16.load_weights(vgg_weight_path, by_name=True)

    # Convolutinalized fully connected layer.
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classifying layers.
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)

    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_3_out)
    block_3_out = BatchNormalization()(block_3_out)

    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_4_out)
    block_4_out = BatchNormalization()(block_4_out)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 8, x.shape[2] * 8)))(x)

    x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model


 ########### Make output image ##############

def make_image(pred):
    img = pred[0]
    seg_img = np.zeros((model_input_height,model_input_width))
    for i in range(0,model_input_height):
        for j in range(0,model_input_width):
            if img[i][j][1]>=img[i][j][0]:
                seg_img[i][j]=255
    
            else:
                seg_img[i][j] = 0

    r = cv2.resize(seg_img,(output_width,output_height))
    return r


###############3 Load model with weights #######################

model = FCN(nClasses,model_input_height,model_input_width,VGG_Weights_path)

#model.load_weights('model_for_original2_weights/Batch_size:20_spe:500epochs:20.5.h5')
#model.load_weights('model_weights_full/Batch_size:20_spe:500epochs:20.7.h5')

model.load_weights('model_weights_full/Re-Batch_size:20_spe:500epochs:20.17.h5')

############## Created and save predicted images ################
# Images are resized and saved in predicted_mask. Then read and predicted. Resized ones are removed then from predicted_mask.

for t in test_images:
        name = t[len(test_folder)+1:].strip(".tiff")
        print(name)
        i1 = cv2.imread(t)
        re = np.float32(cv2.resize(i1,(model_input_width,model_input_height)))/128
        cv2.imwrite("predicted_mask/"+name+".tiff",re)
        i1_re = cv2.imread("predicted_mask/"+name+".tiff")
        i1_re = i1_re.reshape(1,model_input_height,model_input_width,3)
        pred = model.predict(i1_re)
        im = make_image(pred)
        cv2.imwrite(output_folder+"/"+name+".png",im)

######## remove the resized images in the folder ###########
files = glob.glob(output_folder+"/"+"*.tiff")
for f in files:
    os.remove(f)
