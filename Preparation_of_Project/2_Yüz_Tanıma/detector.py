import face_recognition
import pickle
import cv2
import dlib

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector() # yüz tespiti

data = pickle.loads(open('face_rec', "rb").read())


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        for (face, name) in zip(faces, names):
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
            print(name)
            if name ==1:
                cv2.putText(frame, "Berkay", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

            else:
                cv2.putText(frame, "Anonim", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()