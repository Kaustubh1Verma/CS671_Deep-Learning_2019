#!/usr/bin/env python
# coding: utf-8

# In[11] 
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Dropout,UpSampling2D,Input
from tensorflow.python.keras.layers import concatenate as Concatenate
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras import optimizers,regularizers
import cv2


# In[22]:

def auto_encoder(input_shape):
    
    img_input=Input(shape=input_shape)
    x = Conv2D(32,(3,3),padding='same',name="1",activation='relu',input_shape=input_shape,data_format='channels_last')(img_input)  
    x = Conv2D(32,(3,3),padding='same',name="2",activation='relu',data_format='channels_last')(x)
    x = Conv2D(32,(3,3),padding='same',name="3",activation='relu',data_format='channels_last')(x)
    x = Conv2D(32,(3,3),padding='same',name="4",activation='relu',data_format='channels_last')(x) 
    x = Conv2D(32,(3,3),padding='same',name="5",activation='relu',data_format='channels_last')(x)
       
    m = Conv2D(32,(3,3),padding='same',name="6",activation='relu',data_format='channels_last')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Conv2D(64,(3,3),padding='same',name="7",activation='relu',input_shape=input_shape,data_format='channels_last')(x)  
    x = Conv2D(64,(3,3),padding='same',name="8",activation='relu',data_format='channels_last')(x)
    x = Conv2D(64,(3,3),padding='same',name="9",activation='relu',data_format='channels_last')(x)
    x = Conv2D(64,(3,3),padding='same',name="10",activation='relu',data_format='channels_last')(x)
    x = Conv2D(64,(3,3),padding='same',name="11",activation='relu',data_format='channels_last')(x)
    
    b = Conv2D(64,(3,3),padding='same',name="12",activation='relu',data_format='channels_last')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Conv2D(128,(3,3),padding='same',name="13",activation='relu',input_shape=input_shape,data_format='channels_last')(x)  
    x = Conv2D(128,(3,3),padding='same',name="14",activation='relu',data_format='channels_last')(x)
    x = Conv2D(128,(3,3),padding='same',name="15",activation='relu',data_format='channels_last')(x)
    x = Conv2D(128,(3,3),padding='same',name="16",activation='relu',data_format='channels_last')(x)
    x = Conv2D(128,(3,3),padding='same',name="17",activation='relu',data_format='channels_last')(x)
    
    a = Conv2D(128,(3,3),padding='same',name="18",activation='relu',data_format='channels_last')(x)
    
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Conv2D(256,(3,3),padding='same',name="19",activation='relu',input_shape=input_shape,data_format='channels_last')(x) 
    x = Conv2D(256,(3,3),padding='same',name="20",activation='relu',data_format='channels_last')(x)
    x = Conv2D(256,(3,3),padding='same',name="21",activation='relu',data_format='channels_last')(x)
    x = Conv2D(256,(3,3),padding='same',name="22",activation='relu',data_format='channels_last')(x)
    x = Conv2D(256,(3,3),padding='same',name="23",activation='relu',data_format='channels_last')(x)
    
    y = Conv2D(256,(3,3),padding='same',name="24",activation='relu',data_format='channels_last')(x)
    
#     print(y,x)
    
    x = MaxPooling2D(pool_size=2,strides=2)(x)
    x = Conv2D(512,(3,3),padding='same',name="25",activation='relu',input_shape=input_shape,data_format='channels_last')(x)  
    x = Conv2D(512,(3,3),padding='same',name="26",activation='relu',data_format='channels_last')(x)
    x = Conv2D(512,(3,3),padding='same',name="27",activation='relu',data_format='channels_last')(x)
    x = Conv2D(512,(3,3),padding='same',name="28",activation='relu',data_format='channels_last')(x) 
    x = Conv2D(512,(3,3),padding='same',name="29",activation='relu',data_format='channels_last')(x)
    #x = Conv2D(512,(3,3),padding='same',name="30",activation='relu',data_format='channels_last')(x)
    #x = MaxPooling2D(pool_size=2,strides=2)(x)

    
    z = UpSampling2D(size=(2, 2), data_format='channels_last')(x)
    
    c1 = Concatenate([y, z], axis = 3, name = 'c1')
    
    c1 = Conv2D(256,(3,3),padding='same',name="31",activation='relu',input_shape=input_shape,data_format='channels_last')(c1) 
    c1 = Conv2D(256,(3,3),padding='same',name="32",activation='relu',data_format='channels_last')(c1)
    c1 = Conv2D(256,(3,3),padding='same',name="33",activation='relu',data_format='channels_last')(c1)
    c1 = Conv2D(256,(3,3),padding='same',name="34",activation='relu',data_format='channels_last')(c1)
    c1 = Conv2D(256,(3,3),padding='same',name="35",activation='relu',data_format='channels_last')(c1)
    
    z2 = UpSampling2D(size=(2, 2), data_format='channels_last')(c1)
    
    c2 = Concatenate([a, z2], axis = 3, name = 'c2')
    
    c2 = Conv2D(128,(3,3),padding='same',name="36",activation='relu',input_shape=input_shape,data_format='channels_last')(c2) 
    c2 = Conv2D(128,(3,3),padding='same',name="37",activation='relu',data_format='channels_last')(c2)
    c2 = Conv2D(128,(3,3),padding='same',name="38",activation='relu',data_format='channels_last')(c2)
    c2 = Conv2D(128,(3,3),padding='same',name="39",activation='relu',data_format='channels_last')(c2)
    c2 = Conv2D(128,(3,3),padding='same',name="40",activation='relu',data_format='channels_last')(c2)
    
    z3 = UpSampling2D(size=(2, 2), data_format='channels_last')(c2)
    
    c3 = Concatenate([b, z3], axis = 3, name = 'c3')
    
    c3 = Conv2D(64,(3,3),padding='same',name="41",activation='relu',input_shape=input_shape,data_format='channels_last')(c3) 
    c3 = Conv2D(64,(3,3),padding='same',name="42",activation='relu',data_format='channels_last')(c3)
    c3 = Conv2D(64,(3,3),padding='same',name="43",activation='relu',data_format='channels_last')(c3)
    c3 = Conv2D(64,(3,3),padding='same',name="44",activation='relu',data_format='channels_last')(c3)
    c3 = Conv2D(64,(3,3),padding='same',name="45",activation='relu',data_format='channels_last')(c3)
    
    z3 = UpSampling2D(size=(2, 2), data_format='channels_last')(c3)
    
    
    c4 = Concatenate([m, z3], axis = 3, name = 'c4')
    
    c4 = Conv2D(32,(3,3),padding='same',name="46",activation='relu',input_shape=input_shape,data_format='channels_last')(c4) 
    c4 = Conv2D(32,(3,3),padding='same',name="47",activation='relu',data_format='channels_last')(c4)
    c4 = Conv2D(32,(3,3),padding='same',name="48",activation='relu',data_format='channels_last')(c4)
    c4 = Conv2D(32,(3,3),padding='same',name="49",activation='relu',data_format='channels_last')(c4)
    output = Conv2D(3,(3,3),padding='same',name="50",activation='relu',data_format='channels_last')(c4)
    
    model = Model(img_input,output)
    return model






