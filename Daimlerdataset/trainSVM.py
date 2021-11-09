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

#import hog

path = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData"
pos = path + "\\Pedestrians\\48x96"
neg = path + "\\NonPedestrians"

clf = svm.SVC(kernel='linear')
samples = []
labels = []
count = 0

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
for negimg in glob.glob(os.path.join(neg,"*.pgm")):
    img = cv2.imread(negimg)
    img = cv2.resize(img, (48, 96))
    hist = hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm="L1",
               visualize=False, transform_sqrt=False, feature_vector=True)
    labels.append(0)
    hist = np.float32(hist)
    samples.append(hist)
print("done neg")
samples, labels = shuffle(samples, labels)
s_train, s_test, l_train, l_test = train_test_split(samples, labels, test_size=0.1, random_state=109)
clf.fit(s_train,l_train)
l_pred = clf.predict(s_test)
dump(clf, 'trainedmodel.joblib')
print("Accuracy:",metrics.accuracy_score(l_test, l_pred))