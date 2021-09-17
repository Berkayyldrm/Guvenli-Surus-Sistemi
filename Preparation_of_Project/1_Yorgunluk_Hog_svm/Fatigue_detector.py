import pickle
import cv2
import numpy as np
from skimage import feature as ft
import dlib
import time


vid = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = pickle.load(open("model.kayit","rb")) # Kaydedilmiş modeli koda yüklüyoruz.


def eye_status_calculation(array):


    opened = array[-150:].count(1) #son 150 tane frame'i değerlendir(1 olanlanarın sayısını al) (15-10 sn arasındaki frameleri değerlendiriyor fps(pc hızına bağğlı))
    closed = array[-150:].count(0) # tam tersi 0 ları alıyor
    return opened, closed


def Perclos_Calc(opened, closed):
    total_frame = opened + closed
    PERCLOS = closed * 100 / total_frame

    print(f"PERCLOS{PERCLOS}")
    return PERCLOS
array = []
while True:
    _,frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    #yüz
    featureVector=[]

    featureVector2=[]
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
        face_landmarks = dlib_facelandmark(gray, face)

        #sol göz
        u1 = face_landmarks.part(36).x
        u2 = face_landmarks.part(39).x
        o1 = face_landmarks.part(37).y
        o2 = face_landmarks.part(40).y

        lefteye = gray[o1-15:o2+15, u1-15:u2+15]
        cv2.rectangle(img=frame, pt1=(u1-15, o1-15), pt2=(u2+15,o2+15), color=(0, 255, 255), thickness=3)


        resize = cv2.resize(lefteye, (24,24))
        fd, features = ft.hog(resize, orientations=18, pixels_per_cell=(2, 2),
                              cells_per_block=(2, 2), block_norm='L1', visualize=True, transform_sqrt=True,
                              feature_vector=True)
        featureVector.extend(np.array(features).reshape(1, -1))
        b = np.array(model.predict(featureVector))
        array.extend(b)


        cv2.imshow("11", resize)

        #sağ göz
        c1 = face_landmarks.part(42).x
        c2 = face_landmarks.part(45).x
        b1 = face_landmarks.part(43).y
        b2 = face_landmarks.part(46).y

        righteye=gray[b1-15:b2+15,c1-15:c2+15]
        cv2.rectangle(img=frame, pt1=(c1-15, b1-15), pt2=(c2+15,b2+15), color=(0, 255, 255), thickness=3)
        
        
        resize2 = cv2.resize(righteye, (24, 24))
        fd, features2 = ft.hog(resize2, orientations=18, pixels_per_cell=(2, 2),
                               cells_per_block=(2, 2), block_norm='L1', visualize=True, transform_sqrt=True,
                               feature_vector=True)
        featureVector2.extend(np.array(features2).reshape(1, -1))
        a = np.array(model.predict(featureVector2))
        array.extend(a)
        print(array)
        opened,closed=eye_status_calculation(array)

        Perclos_Calc(opened, closed)

        cv2.imshow("12", resize2)


    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
    cv2.imshow("video", frame)

vid.release()
cv2.destroyAllWindows()