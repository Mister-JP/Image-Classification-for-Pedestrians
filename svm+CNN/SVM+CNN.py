import glob
import time
from skimage.color import rgb2gray
import cv2
from rectangle import *
import imutils
import numpy as np
from cv2.cv2 import pyrDown
from joblib import dump, load
from joblib.numpy_pickle_utils import xrange
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import cv2
import numpy as np
from skimage.transform import pyramid_gaussian
import tensorflow as tf


def prediction(img):
    [h, w, d] = (img.shape)
    if h <= 1 or w <= 1:
        return [0]
    img = cv2.resize(img, (48, 96))
    hist = hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm="L1",
               visualize=False, transform_sqrt=False, feature_vector=True)
    hist = np.float32(hist)
    sample = []
    sample.append(hist)
    output = clf.predict(sample)
    return output


def sliding_window(image, stepSize, windowSize):
    for V in range(0, image.shape[0], stepSize):
        for W in range(0, image.shape[1], stepSize):
            yield (W, V, image[V:V + windowSize[1], W:W + windowSize[0]])


start_time = time.time()
Hog = cv2.HOGDescriptor()
Hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
path = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\TestData_1\\03m_28s_762981u.pgm"
#path = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\TestData_1\\00m_25s_577022u.pgm"
path = "C:\\VT\\fall21\\computer vision\\pythonProject\\frames_VT\\frame93.png"
#path = "C:\\VT\\fall21\\computer vision\\pythonProject\\frames_MOT\\1.png"
#path = "C:\\VT\\fall21\\computer vision\\pythonProject\\frames_darsh\\7756.png"
model = tf.keras.models.load_model('fashion_model_dropout_model4_v1.h5py')
fimg = glob.glob(path)
img = cv2.imread(fimg[0])
# cv2.imshow("",img)
clf = load('trainedmodel.joblib')
count = 0
rect = []
downscale = 1.3
window_size = [48, 96]
#check how output of pyramid_gaussian
"""
for (r,i) in enumerate(pyramid_gaussian(img,downscale = downscale)):
    print("shape - ",i.shape)
    cv2.imshow("",i)
    cv2.waitKey(0)
"""

i = img
for (r, i) in enumerate(pyramid_gaussian(img, downscale=downscale)):
    count += 1
    for win in sliding_window(i, 10, window_size):
        if win[2].shape[0] < 96 or win[2].shape[1] < 48:
            continue
        x = win[0]
        y = win[1]
        if prediction(win[2])[0] == 1:
            cv2.imshow("", win[2])
            cv2.waitKey(100)
            temp = []
            im = rgb2gray(win[2])
            image = im.reshape(im.shape[0], im.shape[1], 1)
            image = 255 * image
            temp.append(image)
            temp = np.array(temp)
            print("cnn confidence - ", model.predict(temp)[0])
            if model.predict(temp)[0][0] > 0.99:
                tmp = count
                count = r - 1
                scale = np.flipud(np.divide(i.shape, img.shape))
                app = (x / scale[1], y / scale[2], (x + window_size[0]) / scale[1], (y + window_size[1]) / scale[2])
                rect.append([int(x*(downscale**(count))),int(y*(downscale**(count))),int(x*(downscale**(count))+(window_size[0]/(downscale**(count)))),int(y*(downscale**(count))+(window_size[1]/(downscale**(count))))])
                count = tmp
rect = np.array(rect)
rect = area(rect,0.5)
# print("overlap removed - ",rect)
for i in rect:
    #green box
    cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
print("--- %s seconds ---" % (time.time() - start_time))
#inbuilt peestrian detector
(regions, _) = Hog.detectMultiScale(img, winStride=(4, 4), padding=(4, 4), scale=1.05)
for (x, y, w, h) in regions:
    #red box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("", img)
cv2.waitKey(0)

