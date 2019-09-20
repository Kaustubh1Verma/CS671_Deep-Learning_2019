

import glob
import cv2
import auto_enc
import argparse
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


# Current python dir path
import os
import glob
import argparse
import cv2


dir_path = os.path.dirname(os.path.realpath('__file__'))
parser = argparse.ArgumentParser()
parser.add_argument("-TH", "--test_folder_haze", required=True, help="Folder for the test images")
parser.add_argument("-TG", "--test_folder_gt", required=True, help="Folder for the test images")
parser.add_argument("-O", "--output_folder", required=True, help="Folder for the predicted images")

####### read test images path ####################
args = parser.parse_args()
test_folder_haze = os.path.join(dir_path,args.test_folder_haze)
test_folder_gt = os.path.join(dir_path,args.test_folder_gt)
output_folder = os.path.join(dir_path,args.output_folder)

test_images_haze = glob.glob(test_folder_haze+"/"+"*.jpg")
test_images_gt = glob.glob(test_folder_gt+"/"+"*.jpg")
#print(test_images)


############### Global variables ########################

model_input_height=480
model_input_width=640

output_height = 480
output_width = 640



######################## Create target Directory if don't exist ###########################3
if not os.path.exists(output_folder):
	os.mkdir(output_folder)
	print("Directory " , output_folder ,  " Created ")
else:    
	print("Directory " , output_folder ,  " already exists")





################ Load model with weights #######################

input_shape=(480,640,3)
model = auto_enc.auto_encoder(input_shape)
model.load_weights('AE_weights/Batch_size:1_spe:27310epochs:25.13.h5')

############## Created and save predicted images ################
#ssim_array = []
#psnr_array = []
for t in test_images_haze:
	print(t)
	name = t[len(test_folder_haze):].strip(".jpg")
	print(name)
	i1 = cv2.imread(t)
	i1 = np.reshape(i1,(1,i1.shape[0],i1.shape[1],i1.shape[2]))
	#i_gt = cv2.imread(test_folder_gt+"/"+name+".jpg")
	pred = model.predict(i1)
	cv2.imwrite(output_folder+"/"+name+".jpg",pred[0])

#print(np.mean(ssim_array))
#print(np.mean(psnr_array))



