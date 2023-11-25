import numpy as np
import cv2
import glob
from skimage.feature import hog 
import matplotlib.pyplot as pt
import gradio.components as gd
dance = ["bharatnatyam","kathak","kathakali","kuchipudi","manipuri","mohiniyattam","odissi","sattriya"]
train = []
labels = []
for i in dance:
    for j in glob.glob("C:\project\\train\\"+i+"\\*.jpg"):
        img = cv2.imread(j)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(224,224))
        ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.GaussianBlur(img,(5,5),0)
        img = hog(img)
        train.append(img)
        labels.append(i)


from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()

labels = l.fit_transform(labels)
print(l.classes_)

train = np.float32(train)

print("...Training Started...")
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.5)
svm.setGamma(5)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
#Train the model.
svm.train(train, cv2.ml.ROW_SAMPLE,labels)
print('Finished training process..')


def predict_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(224,224))
    ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = hog(img)
    img = np.float32(img)
    prediction = svm.predict(img.reshape(1,-1))
    return dance[int(prediction[1].ravel()[0])]
    
    
image = gd.Image()
label = gd.Label()

import gradio
gradio.Interface(fn=predict_img,inputs=image,outputs=label).launch(debug=True,share=True)
