import numpy as np
import glob
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm, __all__
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn import metrics
from joblib import dump, load
import imutils

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale, minSize=(50, 100)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

path = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData"
pos = path + "\\Pedestrians\\48x96"
neg = path + "\\NonPedestrians"
retrain = path + "\\retrain"

clf = svm.SVC(kernel='linear')
samples = []
labels = []
count = 0

for retimg in glob.glob(os.path.join(retrain,"*.pgm")):
    img = cv2.imread(retimg)
    print(count)
    count += 1
    hist = hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm="L1",
               visualize=False, transform_sqrt=False, feature_vector=True)
    labels.append(0)
    hist = np.float32(hist)
    samples.append(hist)
print("done retraining")

for negimg in glob.glob(os.path.join(neg,"*.pgm")):
    img = cv2.imread(negimg)
    print(count)
    count += 1
    if count>=2000:
        break
    for img in pyramid(img, 2):
        for im in sliding_window(img, 40, [48,96]):
            if im[2].shape[1]!=48 or im[2].shape[0]!=96:
                continue
            hist = hog(im[2], orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm="L1",
                       visualize=False, transform_sqrt=False, feature_vector=True)
            labels.append(0)
            hist = np.float32(hist)
            samples.append(hist)
print("done neg")

for posimg in glob.glob(os.path.join(pos,"*.pgm")):
    print(count)
    count+=1
    img = cv2.imread(posimg)
    hist = hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm="L1",
               visualize=False, transform_sqrt=False, feature_vector=True)
    labels.append(1)
    hist = np.float32(hist)
    samples.append(hist)
print("done pos")

samples, labels = shuffle(samples, labels)
s_train, s_test, l_train, l_test = train_test_split(samples, labels, test_size=0.1, random_state=109)
clf.fit(s_train,l_train)
l_pred = clf.predict(s_test)
dump(clf, 'trainedmodel.joblib')
print("Accuracy:",metrics.accuracy_score(l_test, l_pred))