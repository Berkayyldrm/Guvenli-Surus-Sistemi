import cv2
import dlib
from scipy.spatial import distance


def calculate_Ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio
def calc_lip(lip):
    A = distance.euclidean(lip[2], lip[10])
    B = distance.euclidean(lip[4], lip[8])
    C = distance.euclidean(lip[0], lip[6])
    lip_ratio = (A + B) / (2.0 * C)
    return lip_ratio

def calculate_last_Ear(left_ear,right_ear):
    EAR = (left_ear + right_ear) / 2
    EAR = round(EAR, 2)
    return EAR

def eye_status_calculation(EAR):
    if EAR < 0.24:
        array.append(0)
    #print(f"ear:{EAR}")
    if EAR >= 0.24:
        array.append(1)

    opened = array[-150:].count(1)
    closed = array[-150:].count(0)
    return opened,closed

def Perclos_Calc(opened,closed):
    total_frame = opened + closed
    PERCLOS = closed * 100 / total_frame
    print(f"PERCLOS{PERCLOS}")

opened=0
closed = 0
total_frame=0
array = []
PERCLOS = 0

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector() # yüz tespiti
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        lip = []

        for n in range(36, 42): #Left Eye
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48): #Right Eye
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(48,60):  #lip
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            lip.append((x, y))
            next_point = n + 1
            if n == 59:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        lip_ratio = calc_lip(lip)
        print(lip_ratio)
        left_ear = calculate_Ear(leftEye)
        right_ear = calculate_Ear(rightEye)
        EAR = calculate_last_Ear(left_ear, right_ear)
        opened, closed = eye_status_calculation(EAR)
        Perclos_Calc(opened,closed)

    cv2.imshow("image", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()