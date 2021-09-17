from imutils import paths
import face_recognition
import pickle
import cv2
import os

imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
IDs = []
counter=0
for (i, imagePath) in enumerate(imagePaths):

    ID = int(os.path.split(imagePath)[-1].split('.')[1])

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=2, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=50)
    counter += 1
    print(f"%{counter}")
    for encoding in encodings:
        knownEncodings.append(encoding)
        IDs.append(ID)
data = {"encodings": knownEncodings, "names": IDs}

f = open("face_rec2", "wb")
f.write(pickle.dumps(data))
f.close()
