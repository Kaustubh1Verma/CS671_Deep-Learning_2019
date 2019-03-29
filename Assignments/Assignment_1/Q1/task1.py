import numpy as np
import cv2
import math 
import random
import os
from tempfile import TemporaryFile
from sklearn.model_selection import train_test_split	
# Creating classes. 
length=[7,15]
width=[1,3]
col=[]
col.append([0,0,255])	#Blue
col.append([255,0,0])	#Red
interval=15
angles=[]
x=0
while x<180:
	angles.append(x)
	x+=interval
dirn=1
a1=0
os.mkdir("/home/aj/Desktop/DL2")	
for l in length:
	a2=0	#a1  0->7,1->15
	for w in width:	
		a3=0				#a2  0->1,1->3	
		for co in col:
			a4=0			#a3  0->red,1->blue
			for ang in angles:
				flag=0
				m=0
				os.mkdir("/home/aj/Desktop/DL2/"+str(dirn))
				while flag<1000:					
					img=np.zeros((28,28,3),np.uint8)
					x=random.randrange((28-math.ceil(l*math.sin(math.radians(180-ang)))))
					y=random.randrange((28-math.ceil(l*math.sin(math.radians(180-ang)))))
					endy = y+l*math.sin(math.radians(180-ang))
					endy=math.floor(endy)
					endx = x+l*math.cos(math.radians(180-ang))
					endx=math.floor(endx)
					if(0<=endx<=28 and 0<=endy<=28):
						cv2.line(img,(x,y),(endx,endy),co,w)
						flag=flag+1
						cv2.imwrite("/home/aj/Desktop/DL2/"+str(dirn)+"/"+str(a1)+"_"+str(a2)+"_"+str(a4)+"_"+str(a3)+"_"+str(flag)+".png",img)	
				dirn+=1
				a4+=1
			a3=a3+1	
		a2=a2+1	
	a1=a1+1
outfile = TemporaryFile()
# Creating Frames 		
train=[]
train_class=[]
test_class=[]
allimg=[]
label=[]
flag=0
# os.mkdir("/home/aj/Desktop/DL2/frames")
for count in range (1,97):	
	f=[]
	# os.mkdir("/home/aj/Desktop/DL2/frames/frame_"+str(count))
	f=os.listdir("/home/aj/Desktop/DL2/"+str(count))
	for fi in f:
		# print(fi)
		n=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+fi)
		n = n.reshape(2352)
		allimg.append(n)
		label.append(flag)
	flag+=1
	for i in range (0,10):
		img1=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i],1)
		img2=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+1],1)
		img3=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+2],1)
		img1f=np.concatenate((img1,img2,img3),axis=1)
		img4=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+3],1)
		img5=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+4],1)
		img6=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+5],1)
		img2f=np.concatenate((img4,img5,img6),axis=1)
		img7=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+6],1)
		img8=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+7],1)
		img9=cv2.imread("/home/aj/Desktop/DL2/"+str(count)+"/"+f[i+8],1)
		img3f=np.concatenate((img7,img8,img9),axis=1)
		imgf=np.concatenate((img1f,img2f,img3f),axis=0)
		cv2.imwrite("/home/aj/Desktop/DL2/frames/frame_"+str(count)+"/"+"f"+str(i+1)+".png",imgf)
# print(allimg[0])
# print(label[0:97])
X_train, X_test, y_oldtrain, y_oldtest = train_test_split(allimg, label, test_size=0.40, random_state=42)
# print(y_oldtrain[0:10])
y_oldtrain = np.array(y_oldtrain).reshape(-1)
y_train=np.eye(96)[y_oldtrain]
y_oldtest = np.array(y_oldtest).reshape(-1)
y_test=np.eye(96)[y_oldtest]
np.savez_compressed("/home/aj/Desktop/DL2/outfile",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
# Creating Video
# img_frame=[]
# for i in range (1,97):
# 	f=[]
# 	f=os.listdir("/home/aj/Desktop/DL2/frames/frame_"+str(i))

# 	path="/home/aj/Desktop/DL2/frames/frame_"+str(i)+"/"
# 	for file in f:
# 		img = cv2.imread(path+file)
# 		height,width,layers = img.shape
# 		size = (width,height)
# 		img_frame.append(img)

# out = cv2.VideoWriter("/home/aj/Desktop/DL2/assign1.mp4",0x7634706d,5, size)
# for i in range(len(img_frame)):
#     out.write(img_frame[i])
# out.release()