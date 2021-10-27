# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
 #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
  #  print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2

#Initiliazing the HOG Person
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Reading the image
image = cv2.imread('pe.jpg')

#Resizing the Image
image = imutils.resize(image, width = min(400,image.shape[1]))

#Detecting all the regions within the image
(regions,_) = hog.detectMultiScale(image, winStride=(4,4),padding=(4,4),scale=1.05)
for(x,y,w,h) in regions:
    print("x = ", x)
    print("y = ", y)
    print("w = ", w)
    print("h = ", h)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

#Showing the image
cv2.imshow("Image",image)
cv2.waitKey(0)

cv2.destroyWindow()



