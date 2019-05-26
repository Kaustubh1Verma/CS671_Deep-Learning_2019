import glob
import cv2
import pickle
import itertools
import numpy as np
from os import listdir
from scipy.io import loadmat
import matplotlib.image as img
from os.path import isfile, join
#from matplotlib import pyplot as plt

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

### using dice coeff for metric
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


##################### Model #################################################

def FCN(num_classes ,  input_height=224, input_width=224,vgg_weight_path=None):

    img_input = Input(shape=(input_height,input_width, 3))

    #img_input = Input(input_shape)

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
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, x)
        vgg16.load_weights(vgg_weight_path, by_name=True)

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

### function to make output for image:
def make_out(im,h,w):
    seg_labels = np.zeros((  h , w  , 2 ))
    
    for i in range(0,224):
        for j in range(0,224):
            if im[i][j]>0:
                seg_labels[i][j][1] = 1
            else:
                seg_labels[i][j][0] = 1
    return seg_labels


############################ Generator to load and train images. ######################################

def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
    
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob( images_path + "*.tiff"  )
    images.sort(key=lambda x: float(x.strip(".tiff").strip(images_path)))

    segmentations = glob.glob( segs_path + "*.tiff"  )
    segmentations.sort(key=lambda x:float(x.strip(".tiff").strip(segs_path)))


    assert len( images ) == len(segmentations)
    zipped = itertools.cycle( zip(images,segmentations) )

    #i=0
    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im,seg = next(zipped)
            #im , seg = zipped.next()
            #print(im)
            
            i1 = cv2.imread(im)
            i2 = make_out(cv2.imread(seg,0),input_height,input_width)

            
            X.append(i1)
            Y.append(i2)

        yield np.array(X) , np.array(Y)



VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


####################### Image parameters ###########################
input_height=224
input_width=224

output_height=224
output_width=224

nClasses = 2


############################# Model summary #################################

# model and its summary
model = FCN(nClasses,input_height,input_width,VGG_Weights_path)
model.summary()


### model compile:
model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])


########################## Load previous model weights(if trained) #######################
#model.load_weights('model_weights_full/Batch_size:20_spe:500epochs:20.7.h5')


########################################################################

## path to load the data
mypath3 = "resized_data/data/"
onlyfiles3 = [f for f in listdir(mypath3) if isfile(join(mypath3, f))]

mypath4 = "resized_data/mask/"
onlyfiles4 = [f for f in listdir(mypath4) if isfile(join(mypath4, f))]


############ Model Specifications ######################################
batch_size = 20
steps_per_epoch = 500
epochs = 20

g= imageSegmentationGenerator(mypath3,mypath4,batch_size,nClasses,input_height,input_width,output_height,output_width)



############## Training of model #################

model_name = "model_weights_full/Re-Batch_size:"+str(batch_size)+"_"+"spe:"+str(steps_per_epoch)+"epochs:"+str(epochs)

for ep in range( epochs ):
    print("Starting Epoch " , ep )
    history = model.fit_generator( g , steps_per_epoch  , epochs=1 )
    if not model_name is None:
        model.save_weights( model_name + "." + str( ep )+".h5")
    print("Finished Epoch" , ep )

    if ep==19 or ep==20:
        ## save history as dictionary
        with open("history_full.pickle", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)





