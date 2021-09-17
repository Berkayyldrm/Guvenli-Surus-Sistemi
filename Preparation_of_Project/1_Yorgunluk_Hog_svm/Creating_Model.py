import os
import matplotlib.image as mpimg
from sklearn.svm import NuSVC
from skimage import feature as ft
import time
import cv2
import numpy as np
import pickle

def loadImgFeature(rootdir):
    allDoc = os.listdir(rootdir) #konumdaki dosyaların isimlerini gösteriyor listeliyor
    classLabelVector = [] #açık yada kapalı olduğunu etiketiyor
    featureVector = []

    for k in range(len(allDoc)):
        doc = os.path.join(rootdir, allDoc[k]) #tüm resimlerin üzerinden geçiyor
        img = mpimg.imread(doc)

        fd, features = ft.hog(img, orientations=18, pixels_per_cell=(2, 2),
                            cells_per_block=(2, 2), block_norm='L1', visualize=True, transform_sqrt=True, feature_vector=True)
        print(features.size)


        featureVector.extend(np.array(features).reshape(1, -1))
        filename = os.path.split(doc)[1]
        listFromLine = str(filename).strip().split('_')
        if listFromLine[0] == 'closed':
            classLabelVector.append(0)
        else:
            classLabelVector.append(1)
    return featureVector, classLabelVector
def opps(model):
    vid = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("frontalface.xml")
    eye_cascade = cv2.CascadeClassifier("Eye_HAAR.xml")

    while True:
        _, frame = vid.read()

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Bulunan yüzleri dikdörtgen içine alıyoruz.

            roi_frame = frame[y:y + h, x:x + w]  # ROİ bölgesini aldık.
            roi_gray = gray[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

                roi2 = roi_gray[ey:ey + eh, ex:ex + ew]
                resize = cv2.resize(roi2, (24, 24))

                print(model.predict(resize))
                cv2.imshow("11", resize)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        cv2.imshow("video", frame)

    vid.release()
    cv2.destroyAllWindows()


start = time.time()
trainX, trainY = loadImgFeature('trainingSet')
testX, testY = loadImgFeature('testSet')
clf = NuSVC()#Modelimiz

clf.fit(trainX, trainY)
opps(clf)
Z = clf.predict(testX)

print("the total error rate is" + str(sum(Z != testY) / float(len(testY))))


error0, error1, total0, total1 = 0, 0, 0, 0

for i in range(len(Z)):
    if testY[i] == 0:

        total0 += 1
        if Z[i] != 0:
            error0 += 1
    else:
        total1 += 1

        if Z[i] != 1:
            error1 += 1
"""print("\nthe total number of positive sample is %d,the positive sample error rate is %f." % (
    total1, error1 / float(total1)))
print("\nthe total number of negative sample is %d,the negative sample error rate is %f." % (
    total0, error0 / float(total0)))
print("spend time:%ss." % (time.time() - start))
"""
