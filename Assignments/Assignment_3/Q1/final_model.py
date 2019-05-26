import tensorflow
import tensorflow.python.keras
# from tensorflow.python.keras.models import load_weights
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.optimizers import RMSprop,adam
from tensorflow.python.keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from IntersectionOverUnion import bb_intersection_over_union
# from dataset.IntersectionOverUnion import bb_intersection_over_union
import numpy as np
import random
import os
import cv2


path_to_data='Q1/outfile.npz'
path_weights = 'weights.h5'

def create_model():
	input=Input(shape=(img_size[0],img_size[1],1))
	x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(input)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Conv2D(32,(3,3),use_bias=False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Conv2D(64,(3,3),use_bias= False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D((2,2))(x)
	x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Conv2D(128,(3,3),use_bias= False, padding = 'same')(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D((2,2))(x)
	features = Flatten()(x)
	x = Dense(512,use_bias=False)(features)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Dense(128,use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Dense(64,use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Dense(32,use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = Dense(32,use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	output_class = Dense(3,activation = 'softmax')(x)
	output_regress = Dense(4)(x)
	network = Model(input,[output_class,output_regress])
	return network


def load_trainedmodel(path,X_test):
	new_model=create_model()
	new_model.summary()
	new_model.load_weights(path)
	pred = new_model.predict(X_test)
	return pred


def distribute_data(data):
	a=[];b=[]
	for grd in data:
		a.append(int(grd[0])-1)
		b.append([grd[1],grd[2],grd[3],grd[4]])	
	a=np.array(a).reshape(-1)
	a=np.eye(3)[a]
	b=np.array(b)
	return a,b	

def save(file,Truth,Predicted):
	with open(file, 'a+') as f:
		f.write("Truth :")
		for item in Truth:
			f.write("%s " % item)
		f.write("Predicted: ")
		for item2 in Predicted:
			f.write("%s " % item2)
		f.write("IOU: ")
		f.write(str(bb_intersection_over_union(Truth,Predicted)))

	
def load_data(path):
	data = np.load(path)
	X_train=data["X_train.npy"]
	X_test=data["X_test.npy"]
	y_train=data["y_train.npy"]
	y_test=data["y_test.npy"]
	y_trainclass,y_trainregress=distribute_data(y_train)
	y_testclass,y_testregress=distribute_data(y_test)
	return X_train,X_test,y_trainclass,y_trainregress,y_testclass,y_testregress

X_train,X_test,y_trainclass,y_trainregress,y_testclass,y_testregress=load_data(path_to_data)
img_size=X_train[0].shape
X_test = X_test.reshape(X_test.shape[0],img_size[0],img_size[1],1).astype('float32')
img_size=X_train[0].shape
X_predicted=load_trainedmodel(path_weights,X_test)

for i in range(10):
	print("Actual Label")
	print(y_testregress[i])

for i in range(10):
	print("Predicted Label")
	print(X_predicted[1][i])

for i in range(0,len(y_testregress)):
	file = open('Output.txt', 'w+')
	save('Output.txt',y_testregress[i],X_predicted[1][i])
	file.close()	

