import os

import cv2
vidcap = cv2.VideoCapture('MOT.webm')
success,image = vidcap.read()
count = 0
while success:
  path = 'C:\\VT\\fall21\\computer vision\\pythonProject\\frames_MOT'
  cv2.imwrite(os.path.join(path, "%d.png" % count), image)
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1