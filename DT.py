import numpy as np 
import pandas as pd 
import os
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
            img = cv2.imread(subject+'/'+j+'/'+k,0)
            img=cv2.resize(img,(int(160),int(60)))
            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            imgD=np.asarray(img,dtype=np.float64)
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
    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray)
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
x_train,x_further,y_train,y_further = train_test_split(x_data,y,test_size = 0.2)
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(x_train)
X_train = scaler.transform(x_train)  
X_test = scaler.transform(x_further)  
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X_train, y_train)
y_pred_dt=clf.predict(X_test)
y_train_score_dt=clf.predict(X_train)
from sklearn.metrics import accuracy_score
print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_dt, normalize=True, sample_weight=None))
print('Train',accuracy_score(y_train, y_train_score_dt, normalize=True, sample_weight=None))