import cv2
import sys
cpt=0

vidStream = cv2.VideoCapture(0)
while True:
    ret, frame = vidStream.read() # read frame and return code
    
    cv2.imshow("Test Window",frame) # Show image in the window

    cv2.imwrite(r'D:\Python\Projects -Incomplete\Face Recognition\Images\1\image%04i.jpg' %cpt, frame) # To store the images

    cpt +=1

    if cv2.waitKey(10)==ord('q'):
        break