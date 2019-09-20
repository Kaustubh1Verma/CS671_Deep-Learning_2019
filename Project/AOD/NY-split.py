import glob
import cv2, numpy as np
from sklearn.model_selection import train_test_split


def read_image(path):
	return cv2.imread(path)

def save_image(path_X,path_Y,X,Y):
	for i in range(len(X)):
		cv2.imwrite(path_X+str(i)+".jpg",read_image(X[i]))
		cv2.imwrite(path_Y+str(i)+".jpg",read_image(Y[i]))


train_data_dir = 'train_haze/'
train_label_dir = 'train_gt/'

train_path = glob.glob(train_data_dir+'/'+'*.jpg')
label_path = glob.glob(train_label_dir+'/'+'*.jpg')


train_X, test_X,train_Y, test_Y = train_test_split(train_path,label_path,test_size=0.10,shuffle=True)

### save split images ####
path_train_X = "NYU/train/haze/" 
path_train_Y = "NYU/train/gt/"
path_test_X = "NYU/test/haze/" 
path_test_Y = "NYU/test/gt/"

save_image(path_train_X,path_train_Y,train_X,train_Y)
save_image(path_test_X,path_test_Y,test_X,test_Y)
