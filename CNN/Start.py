import glob
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

path = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData"
pos = path+"\\dummyp"
neg = path + "\\dummynp"

def dataprep(pos,neg):
    data = [pos,neg]
    label = [[1,0],[0,1]]
    label = np.float32(label)
    train_x=[]
    train_y = []
    for pat in range(len(data)):
        print("strating loading data from - ",data[pat])
        for img in glob.glob(os.path.join(data[pat],'*.pgm')):
            image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            image = image.reshape(image.shape[0],image.shape[1],1)
            image = image/255
            train_x.append(image)
            train_y.append(label[pat])
        print("loaded data from ", data[pat])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_test_split(train_x, train_y, test_size=0.2, random_state=13)
    train_x,valid_x,train_label,valid_label = train_test_split(train_x, train_y, test_size=0.2, random_state=13)
