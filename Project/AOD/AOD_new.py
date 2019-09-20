import scipy, pickle ,re
import tensorflow as tf
from scipy import ndimage
from scipy.misc import imsave
import tensorflow.keras as keras
import matplotlib.image as plt_img
import os,cv2,glob,itertools,numpy as np,math as m
from sklearn.model_selection import train_test_split
#from skimage.measure import psnr,ssim
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.activations import relu 
from tensorflow.python.keras import backend as K,optimizers
from tensorflow.python.keras.models import Sequential, load_model,Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Lambda,Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose 

#### Image height and width ##########
img_width, img_height = 640, 480


####################### Fucntions ########################################3
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


def get_unet(arg1, height , width ,channel , trainable = True ):
    inputs = Input(shape=(height,width, channel))
    conv1 = Conv2D(3, (1, 1), kernel_initializer='random_normal', activation='relu', trainable = trainable)(inputs)

    conv2 = Conv2D(3, (3, 3), kernel_initializer='random_normal', activation='relu',  padding='same', trainable = trainable)(conv1)

    concat1 = concatenate([conv1, conv2], axis=-1)

    conv3 = Conv2D(3, (5, 5), activation='relu', kernel_initializer='truncated_normal', padding='same', trainable = trainable)(concat1)

    concat2 = concatenate([conv2, conv3], axis=-1)

    conv4 = Conv2D(3, (7, 7), activation='relu', kernel_initializer='random_normal', padding='same', trainable = trainable)(concat2)

    concat3 = concatenate([conv1, conv2, conv3, conv4], axis=-1)

    K = Conv2D(3, (3, 3), activation='relu', kernel_initializer='truncated_normal', padding='same', trainable = True)(concat3)

    print(inputs.shape,K.shape)
    product= keras.layers.Multiply()([K, inputs])
    sum1 = keras.layers.Subtract()([product, K])
    sum2 = Lambda(lambda x: 1+x) (sum1)
    out_layer = Lambda(lambda x: relu(x)) (sum2)

    if arg1 == 1:
        model = Model(inputs=inputs,outputs=out_layer)
    else:
        model = Model(inputs=inputs,outputs=conv1)

    return model



def load_test(test_path,test_label):
	x=[]
	y=[]
	for i in range(len(test_path)):
		x.append(cv2.imread(test_path[i]))
		y.append(cv2.imread(test_label[i]))
	return  np.array(x),np.array(y)	


def avg_ssim_psnr(val_x,val_y):

    for i in range(len(val_x)):
        tx = val_x[i]
        ty = val_y[i]
	
        tx = np.reshape(tx,(1,tx.shape[0],tx.shape[1],tx.shape[2]))
	
        gt.append(ty)
        predicted.append(model.predict(tx)[0])

    l=len(gt)
    avg_ssim=0
    avg_psnr=0
    print("xyz")

    for i in range(l):
        if i%50==0:
            print("kuch bhi")  
        a_tf = tf.convert_to_tensor(predicted[i],np.uint8)
        b_tf = tf.convert_to_tensor(gt[i],np.uint8)
        a=tf.image.ssim(a_tf,b_tf,max_val=255)
        b=tf.image.psnr(a_tf,b_tf,max_val=255)  
        avg_ssim=avg_ssim+k.eval(a)
        avg_psnr=avg_psnr+k.eval(b)
    return avg_ssim/l,avg_psnr/l



########################################### READING TRAINING DATA ################################333
train_data_dir = 'NYU/train/haze/'
train_label_dir = 'NYU/train/gt/'

test_data_dir = 'NYU/test/haze/'
test_label_dir = 'NYU/test/gt/'


batch_size = 1
    
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)

model = get_unet(1,480,640,3, True)

######## Optimizers ########
opt = optimizers.RMSprop(lr=0.001, decay=0.0001, rho=0.9)
#opt = optimizers.Adam(lr=0.001,decay=0.0001)

model.compile(loss='mean_squared_error',optimizer=opt,metrics=['accuracy'])
model.summary()  


##################	MODEL SPECIFICATIONS #########################

epochs = 25
train_path = glob.glob(train_data_dir+'/'+'*.jpg')
label_path = glob.glob(train_label_dir+'/'+'*.jpg')

test_X = glob.glob(test_data_dir+'/'+'*.jpg')
test_Y = glob.glob(test_label_dir+'/'+'*.jpg')


###############################################33


train_X,_,train_Y, _ = train_test_split(train_path,label_path,test_size=0,shuffle=True)

steps_per_epoch = int(len(train_X)/batch_size)
train_set_size = len(train_X)
print("Train set size is:", train_set_size, ",", "Steps per epochs is:", steps_per_epoch, ",", "batch size is:", batch_size, "total epochs is:", epochs)

validation_steps = int(len(test_X)/batch_size)
validation_size = len(test_X)
print("Validation size:",validation_size,"Validation steps:", validation_steps)

g=train_generator(train_X,train_Y,batch_size)

######################    Loading Test Data  ########################33
val_x,val_y=load_test(test_X, test_Y)


model_name = "AOD_weights/Batch_size:"+str(batch_size)+"_"+"spe:"+str(steps_per_epoch)+"epochs:"+str(epochs)


history_list = []

for ep in range( epochs ):
    print("Starting Epoch" , ep )
    history=model.fit_generator(generator=g,steps_per_epoch=steps_per_epoch, validation_data=(val_x,val_y),epochs=1,verbose=1)
    if not model_name is None:
        model.save_weights( model_name + "." + str( ep )+".h5")
    print("Finished Epoch" , ep )

    history_list.append(history.history)

    ##### validate and get average psnr and ssim ##########
    predicted = []
    gt = []
    
    ssim,psnr = avg_ssim_psnr(val_x[0:100],val_y[0:100])
    print("The average SSIM and PSNR is ",ssim," and ",psnr)   	
	
    if ep%5==0:
        with open("AOD_history/AOD_history_list"+str(ep)+"_.pickle", 'wb') as file_pi:
            pickle.dump(history_list, file_pi) 
