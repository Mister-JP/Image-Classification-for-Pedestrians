import glob
import os
import numpy as np
from joblib import dump, load
import cv2.cv2
from skimage.feature import hog
import readsequence


seq_separator =":"
img_separator = ";"
obj_2d_separator = "#"


class reader:
    def prediction(self,img):
        [h,w,d] = (img.shape)
        if h<=1 or w<=1:
            return 0
        img = cv2.resize(img, (48, 96))
        hist = hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm="L1",
               visualize=False, transform_sqrt=False, feature_vector=True)
        hist = np.float32(hist)
        sample = []
        sample.append(hist)
        output = self.clf.predict(sample)
        return output
    def __init__(self):
        self.seqcount = -1
        self.imgcount = -1
        self.path = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\TestData_1"
        self.flag = 0
        self.clf = load('trainedmodel.joblib')

        self.seqs = []

        f = open("GroundTruth2D.db", "r")
        #Lines = f.readline()

        line = f.readline()
        line = line.strip()
        while line:
            #print("here")
            if line==":":
                seq_id = f.readline().strip()
                path_to_data = f.readline().strip()
                numimages = f.readline().strip()
                seq = readsequence.readSequences(seq_id, path_to_data, numimages)
                self.seqs.append(seq)
                self.seqcount+=1
            elif line==";":
                self.imgcount+=1
                nameimg = f.readline().strip()
                [width,height] = f.readline().strip().split(" ")
                if self.flag!=0:
                    cv2.imshow("",self.img)
                    cv2.waitKey(5)
                fimg = glob.glob(os.path.join(self.path, nameimg))
                self.img = cv2.imread(fimg[0])
                self.flag=1
                numobj = f.readline().strip().split(" ")[1]
                self.seqs[self.seqcount].addimage(nameimg, width, height, numobj)
            elif line[0] =="#":
                objclass = line[2:].strip()
                objid = f.readline().strip()
                conf = f.readline().strip()
                [min_x,min_y,max_x,max_y] = f.readline().strip().split(" ")
                cropped = self.img[int(min_y):int(max_y), int(min_x):int(max_x)]
                if self.prediction(cropped)==1:
                    cv2.rectangle(self.img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
                else:
                    cv2.rectangle(self.img, (int(min_x),int(min_y)), (int(max_x),int(max_y)), (0,0,255),2)
                self.seqs[self.seqcount].addobj(objclass, objid, conf, [min_x,min_y, max_x,max_y])
            line = f.readline()
            line=line.strip()