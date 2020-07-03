import numpy as np
import cv2
import os

import face_recognition as fr
print(fr)

test_img=cv2 .imread(r'D:\Python\Projects\Face Recognition\test_image5.jfif') # Give path to the image which you want to test

faces_detected,gray_img = fr.faceDetection(test_img)
print('Face Detected: ',faces_detected)

#Training will begin from here

faces,faceID = fr.labels_for_training_data("D:\Python\Projects\Face Recognition\Image")
face_recogniser = fr.train_Classifier(faces,faceID)
face_recogniser.save('D:/Python/Projects/Face Recognition/TrainingData.yml') # To save the trained model. Just give the path.
# Assign labels to images folder
name={0:"Rhythem Jain",1:"Mark Zuckerberg", 2: "Hrithik Roshan", 3:"Deepika Padukone", 4: "Emma Watson"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence = face_recogniser.predict(roi_gray)
    print("Label: ",label) # Label is 0, 1 assigned to the names
    print("Confidence: ",confidence)
    fr.draw_rect(test_img,face)
    predict_name=name[label] #labels used to get the names 
    fr.put_text(test_img,predict_name,x,y)

resized_img = cv2.resize(test_img,(1000,700))

cv2.imshow("Face Detection",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
