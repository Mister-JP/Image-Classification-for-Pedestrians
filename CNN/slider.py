import glob
import os
import cv2
import numpy as np
from skimage.transform import pyramid_gaussian
import tensorflow as tf
from rectangle import *




def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def slider(path, mdl):
    test = []
    temp=[]
    coord = []
    window_size = [48,96]
    downscale=1.2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for (r, i) in enumerate(pyramid_gaussian(img, downscale=downscale)):
        for im in sliding_window(i, 10, [48,96]):
            if im[2].shape[0]!=96 or im[2].shape[1]!=48:
                continue
            #cv2.imshow("",im[2])
            #cv2.waitKey(10)
            image = im[2].reshape(im[2].shape[0], im[2].shape[1], 1)
            #print(image)
            #image=image
            test.append(image)
            temp.append(image)
            temp=np.array(temp)
            if mdl.predict(temp)[0][0]>0.999:
                print("confidence - ",mdl.predict(temp)[0][0])
                cv2.imshow("",im[2])
                cv2.waitKey(10)
            temp=[]
            coord.append([im[0],im[1], i.shape])
    test = np.array(test)
    #img = img.reshape(img.shape[0], img.shape[1], 1)
    #img = img/255
    #test.append(img)
    #test = np.array(test)
    pred = mdl.predict(test)
    print("pred - ",pred)
    rect=[]
    for i in range(len(pred)):
        #print("pred[i] = ",pred[i])
        if pred[i][0]>0.99:
            x = coord[i][0]
            y = coord[i][1]
            scale = np.flipud(np.divide(coord[i][2], img.shape))
            print("scale - ",scale)
            app = (x / scale[0], y / scale[1], (x + window_size[0]) / scale[0], (y + window_size[1]) / scale[1])
            rect.append(app)
    print("len of rect = ",len(rect))
    rect = area(rect,0.5)
    for i in rect:
        cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
    return img

#testing
path = path = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\TestData_1"
path = path+"\\00m_25s_577022u.pgm"
path = "C:\\VT\\fall21\\computer vision\\pythonProject\\frames\\frame345.png"
model = tf.keras.models.load_model('fashion_model_dropout_model4.h5py')
rec = slider(path, model)
cv2.imshow("",rec)
cv2.waitKey()
pathpos = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData\\Pedestrians\\48x96\\pos10355.pgm"
pathneg = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData\\extractednp\\frame9878.pgm"
print("for neg")
negrecieved = slider(pathneg, model)
print("for pos")
posrecieved = slider(pathpos, model)


cv2.imshow("", recieved)
cv2.waitKey(0)