import numpy as np
import cv2 # To convert any image into pixel form 
import os


def faceDetection(test_img): 
    # Going to do face recognition on greyscale
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY) #Convert color image to gray
    face_haar = cv2.CascadeClassifier(r'C:\Python\Python38\Scripts\haarcascade_frontalface_alt.xml') #To detect the face from the image
    faces = face_haar.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=3) #scale factor to remove side part other than face
    return faces,gray_img

def labels_for_training_data(directory): # 0 is the label and the images are filenames, Images is the directory
    faces=[]
    faceID=[]
    for path,subdirnames,filenames in os.walk(directory): # 0 is the label and and images in it are the filenames 
        for filename in filenames:
            if filename.startwith('.'):
                print('Skipping system file') # if anything starts with not i(image) then  will not read it
                continue # To skip the error
            id = os.path.basename(path)
            img_path = os.path.join(path,filename) # finding the path of image for line test_img in line 24
            print('img_path',img_path)
            print('id',id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print('Not Loaded Properly') # for error
                continue

            # If images are there
            faces_rect, gray_img = faceDetection(test_img)
            (x,y,w,h)=faces_rect[0] # making the rectangle on the face
            roi_gray=gray_img[y:y+w,x:x+h] # roi is reason of interest(part required)
            faces.append(roi_gray)
            faceID.append(int(id))
        return faces, faceID

def train_Classifier(faces, faceID):# Giving the face with it's ID to tell ki agar aisa chehra hai toh ye face id hai
    face_recogniser = cv2.face.LBPHFaceRecognizer_create()
    face_recogniser.train(faces,np.array(faceID)) # converted to np.array to check from the ids of different people
    return face_recogniser

def draw_rect(test_img, face):
    (x,y,w,h) = face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

def put_text(test_img,label_name,x,y):
    cv2.putText(test_img,label_name,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),3)


    