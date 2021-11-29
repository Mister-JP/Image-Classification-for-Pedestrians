import os
import glob
import time
from skimage.transform import pyramid_gaussian
import numpy as np
from rectangle import *
import cv2.cv2
import imutils
from joblib.numpy_pickle_utils import xrange
from skimage.feature import hog
from joblib import dump, load
import tensorflow as tf
from skimage.color import rgb2gray

start_time = time.time()
path = "C:\\VT\\fall21\\computer vision\\diamler\\DaimlerBenchmark\\Data\\TestData"
path = "C:\\VT\\fall21\\computer vision\\pythonProject\\frames_VT"
#path="C:\\VT\\fall21\\computer vision\\pythonProject\\frames"
#model = tf.keras.models.load_model('fashion_model_dropout_model4_v1.h5py')
model = tf.keras.models.load_model('fashion_model_dropout_model4_v1.h5py')
Hog = cv2.HOGDescriptor()
Hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
count=0
img_name = 0
npath = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\output2 model v4"
npath = "C:\\VT\\fall21\\computer vision\\pythonProject\\o_frames_VT5"


import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def non_max_suppression_slow(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in xrange(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[j]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
        return boxes[pick]

def prediction(img):
    dim = (img.shape)
    if dim[0] <= 1 or dim[1] <= 1:
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
            yield [W, V, image[V:V + windowSize[1], W:W + windowSize[0]]]

clf = load('trainedmodel.joblib')
window_size = [48,96]
downscale = 1.3
for im in sorted(glob.glob(os.path.join(path, "*.png")), key = numericalSort):
    img = cv2.imread(im)
    count=0
    rect = []
    for (r, i) in enumerate(pyramid_gaussian(img, downscale=downscale)):
        count += 1
        for win in sliding_window(i, 10, window_size):
            if win[2].shape[0] != window_size[1] or win[2].shape[1] != window_size[0]:
                continue
            win[2] = cv2.resize(win[2], dsize=(48, 96), interpolation=cv2.INTER_CUBIC)
            x = win[0]
            y = win[1]
            win[2] = rgb2gray(win[2])
            if prediction(win[2])[0] == 1:
                temp = []
                im = win[2]
                image = im.reshape(im.shape[0], im.shape[1], 1)
                image = 255 * image
                temp.append(image)
                temp = np.array(temp)
                print("cnn confidence - ", model.predict(temp)[0])
                cv2.imshow("", win[2])
                cv2.waitKey(100)
                if model.predict(temp)[0][0] > 0.999:
                    tmp = count
                    count = r - 1
                    scale = np.flipud(np.divide(i.shape, img.shape))
                    app = (x / scale[1], y / scale[2], (x + window_size[0]) / scale[1], (y + window_size[1]) / scale[2])
                    rect.append(app)
                    count = tmp
    rect = np.array(rect)
    rect = area(rect,0.5)
    for i in rect:
        cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
    print("--- %s seconds ---" % (time.time() - start_time))
    (regions, _) = Hog.detectMultiScale(img, winStride=(4, 4), padding=(4, 4), scale=1.05)
    for (x, y, w, h) in regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("", img)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(npath, "frame%d.png" % img_name), img)
    img_name += 1
    print("image written")
