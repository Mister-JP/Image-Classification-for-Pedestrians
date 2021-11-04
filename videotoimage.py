import cv2
import imutils


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('pedestrian.mp4')
while cap.isOpened():
    ret,image = cap.read()
    if ret:
        image = imutils.resize(
            image, width = min(400, image.shape[1])
        )
        (regions, _) = hog.detectMultiScale(image,
                                           winStride = (4,4),
                                           padding = (4,4),
                                           scale = 1.05)
        flag=0
        for (x,y,w,h) in regions:
            flag=1
            cv2.rectangle(image, (x,y), (x+w, y+h),(0,0,255),2)
        if flag==1 or flag==0:
            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF ==ord('q'):
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()