#!/usr/local/bin/python3

import cv2
import argparse
import os
import glob

# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-ext", "--extension", required=False, default='pgm', help="extension name. default is 'png'.")
#ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
#args = vars(ap.parse_args())

# Arguments
dir_path = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\output t1"
#ext = args['.pgm']
#output = args['output']
ext = ".pgm"
output = "output.mp4"

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

#for image in glob.glob(os.path.join(dir_path,"*.pgm")):
for i in range(0,3535):
    s = str(i)
    image = "C:\\VT\\fall21\\computer vision\\diamler test data\\DaimlerBenchmark\\Data\\output t1\\frame"+s+".pgm"
    #image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))