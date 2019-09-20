import auto_enc
import scipy, pickle ,re
from scipy import ndimage
from scipy.misc import imsave
import tensorflow.keras as keras
import matplotlib.image as plt_img
import os,cv2,glob,itertools,numpy as np,math as m
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras import backend as K,optimizers
from tensorflow.python.keras.models import Sequential, load_model,Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Lambda,Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose


###################### Other imports ################################3
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam


def get_image(path):
    return np.expand_dims(cv2.imread(path[0]), axis = 0)

def train_generator(train_path, label_path, batch_size):
    L = len(train_path)

    #this line is just to make the generator infinite, keras needs that    
    while True:
        batch_start = 0
        batch_end = batch_size
        
        zipped=itertools.cycle(zip(train_path,label_path))
        
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = next(zipped) 

            im = cv2.imread(im )
            seg = cv2.imread(seg)

            X.append(im)
            Y.append(seg)

        yield np.array(X) , np.array(Y)


def load_test(test_path,test_label):
	x=[]
	y=[]
	for i in range(len(test_path)):
		x.append(cv2.imread(test_path[i]))
		y.append(cv2.imread(test_label[i]))
	return  np.array(x),np.array(y)	


############ READING TRAINING DATA ######################
train_data_dir = '../AOD/train_haze/'
train_label_dir = '../AOD/train_gt/'
batch_size = 1

input_shape=(480,640,3)
model = auto_enc.auto_encoder(input_shape)    
opt = adam(lr=0.0001)
model.compile(loss='mean_squared_error',optimizer=opt)
model.summary()


############	MODEL SPECIFICATION ##################

epochs = 25
train_path = glob.glob(train_data_dir+'/'+'*.jpg')
label_path = glob.glob(train_label_dir+'/'+'*.jpg')
train_X, test_X,train_Y, test_Y = train_test_split(train_path,label_path,test_size=0.05,shuffle=True)
steps_per_epoch = int(len(train_X)/batch_size)
train_set_size = len(train_X)
print("Train set size is:", train_set_size, ",", "Steps per epochs is:", steps_per_epoch, ",", "batch size is:", batch_size, "total epochs is:", epochs)
validation_steps = int(len(test_X)/batch_size)
print("Validation steps:", validation_steps)
g=train_generator(train_X,train_Y,batch_size)

################33    Loading Test Data ##############################
val_x,val_y=load_test(test_X, test_Y)


############################ Fit and save model ############################
model_name = "AE_weights/Batch_size:"+str(batch_size)+"_"+"spe:"+str(steps_per_epoch)+"epochs:"+str(epochs)

for ep in range( epochs ):
    print("Starting Epoch" , ep )
    history=model.fit_generator(generator=g,steps_per_epoch=steps_per_epoch, validation_data=(val_x,val_y),epochs=1,verbose=1)
    if not model_name is None:
        model.save_weights( model_name + "." + str( ep )+".h5")
    print("Finished Epoch" , ep )

    if ep==24 or ep==25:
        with open("AE_weights/AE_history.pickle", 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

