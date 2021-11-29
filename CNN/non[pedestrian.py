import glob
import os
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian

path = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData"
neg = path +"\\NonPedestrians"
npath = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData\\extractednp1"
img_name=0
count =0

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

for img in glob.glob(os.path.join(neg,'*.pgm')):
    img = cv2.imread(img)
    print("count- ",count)
    count+=1
    downscale = 1.3
    for (i, image) in enumerate(pyramid_gaussian(img, downscale=downscale)):
        for win in sliding_window(image, [48, 96], [48,96]):
            if win[2].shape[0]!=96 or win[2].shape[1]!=48:
                continue
            gray = rgb2gray(win[2])
            gray = 255 * gray
            cv2.imwrite(os.path.join(npath, "frame%d.pgm" % img_name), gray)
            img_name+=1
            if img_name==30000:
                exit()