import numpy as np 
import pandas as pd 
import os
import time
print(os.listdir("C://Users//DELL//Downloads//Practice2//dataset//dataset//training_set"))
from PIL import Image
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
path="C://Users//DELL//Downloads//Practice2//dataset//dataset//training_set"
folders=os.listdir(path)
folders=set(folders)
different_classes=os.listdir(path)
different_classes=set(different_classes)
print("The different classes that exist in this dataset are:")
print(different_classes)
x=[]
z=[]
y=[]#converting the image to black and white
threshold=200
import cv2
for i in folders:
    print('***',i,'***')
    subject=path+'/'+i
    subdir=os.listdir(subject)
    subdir=set(subdir)
    for j in subdir:
        #print(j)
        images=os.listdir(subject+'/'+j)
        for k in images:
            results=dict()
            results['y']=j.split('_')[0]
            img = cv2.imread(subject+'/'+j+'/'+k,0) # Read the image in gray scale mode with possible noise reduction
            img=cv2.resize(img,(int(160),int(60)))
            
            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Histogram with 2 peaks a good threshold would be in middle of those values

            imgD=np.asarray(img,dtype=np.float64) # Convert PIL images into numpy Array
            z.append(imgD)
            imgf=np.asarray(imgf,dtype=np.float64)
            x.append(imgf)
            y.append(int(j.split('_')[0]))
            results['x']=imgf
l = []
list_names = []
for i in range(6):
    l.append(0)
for i in range(len(x)):
    if(l[y[i] - 1] == 0):
        l[y[i] - 1] = i
        if(len(np.unique(l)) == 6):
            break
for i in range(len(l)):
   
    print("Class Label: " + str(i + 1))
    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray) # Show the image in the newly created image window
    plt.show()
    plt.imshow(np.asarray(x[l[i]]), cmap = cm.gray)     
    plt.show()
x=np.array(x)
y=np.array(y)
y = y.reshape(len(x), 1)
print(x.shape)
print(y.shape)
print(max(y),min(y))
x_data = x.reshape((len(x), 60, 160, 1))

x_data/=255
x_data=list(x_data)
for i in range(len(x_data)):
    x_data[i]=x_data[i].flatten()
print(len(x_data))
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
x_data=np.array(x_data)
print("Before PCA",x_data.shape)
x_data=pca.fit_transform(x_data)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  

print('___________________')
print("After PCA",x_data.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.25, random_state=0) # Split the training set into 25% and 75% sets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 1))
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
pre=knn.predict(X_train)
from sklearn.metrics import accuracy_score
print('KNN Accuracy for test_dataset: %.3f' % accuracy_score(y_test,predictions))
print('KNN Accuracy for training_dataset: %.3f' % accuracy_score(y_train,pre))

