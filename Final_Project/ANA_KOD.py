from sys import argv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from csv import DictReader, DictWriter
from scipy.spatial import distance
from face_recognition import face_locations, face_encodings, compare_faces
from pickle import dumps, loads
from dlib import get_frontal_face_detector, shape_predictor
from keras.models import load_model
from time import time, sleep
from imutils import paths
import os
import cv2
import numpy as np
#import serial

#arduino = serial.Serial(port='COM4', baudrate=9600, timeout=0.1)


class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi("1.ui", self)
        self.Giris.clicked.connect(self.login_function)
        self.Yeni_Kayit.clicked.connect(self.go_to_create)
        self.Sifre_Line.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pushButton.clicked.connect(self.learnpassword)

    def login_function(self):
        username = self.Kullanici_Adi_Line.text()
        password = self.Sifre_Line.text()
        with open('Logins.csv', newline='') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                if row['username'] == username and row["password"] == password:
                    self.textBrowser.setText("Giriş Başarılı")
                    recog = Recognition()
                    widget.addWidget(recog)
                    widget.setCurrentIndex(widget.currentIndex() + 1)
                    break

                else:
                    with open('AdminLog.csv', newline='') as csvfile2:
                        reader = DictReader(csvfile2)
                        for row2 in reader:
                            if row2['username'] == username and row2["password"] == password:
                                self.textBrowser.setText("Admin Girişi Başarılı")
                                adminrecog = AdminRecognition()
                                widget.addWidget(adminrecog)
                                widget.setCurrentIndex(widget.currentIndex() + 1)
                                break

                            else:
                                self.textBrowser.setText("Kullanıcı adı veya şifre yanlış. Tekrar Deneyin.")
                                break
                    self.textBrowser.setText("Kullanıcı adı veya şifre yanlış. Tekrar Deneyin.")

    @staticmethod
    def go_to_create():
        createacc = CreateAcc()
        widget.addWidget(createacc)
        widget.setCurrentIndex(widget.currentIndex()+1)

    @staticmethod
    def learnpassword():
        psw = PasswordScreen()
        widget.addWidget(psw)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class PasswordScreen(QDialog):
    def __init__(self):
        super(PasswordScreen, self).__init__()
        loadUi("Sifre.ui", self)
        self.pushButton.clicked.connect(self.learn)
        self.pushButton_2.clicked.connect(self.back)

    @staticmethod
    def back():
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def learn(self):
        module_number = self.lineEdit.text()
        username = self.lineEdit_2.text()
        with open('Logins.csv', newline='') as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                if row['username'] == username and row["module_number"] == module_number:
                    self.textBrowser.setText(f"Sifreniz = {row['password']}")
                    break
                else:
                    self.textBrowser.setText("Kullanıcı adı veya Modül no yanlış. Tekrar Deneyin.")


class CreateAcc(QDialog):
    def __init__(self):
        super(CreateAcc, self).__init__()
        loadUi("2.ui", self)
        self.pushButton.clicked.connect(self.createaccfunction)
        self.lineEdit_3.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_4.setEchoMode(QtWidgets.QLineEdit.Password)
        self.pushButton2.clicked.connect(self.back)

    @staticmethod
    def back():
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def createaccfunction(self):
        module_number = self.lineEdit.text()
        if module_number == "0xFF" or "0xFE":
            username = self.lineEdit_2.text()
            password = self.lineEdit_3.text()
            confirm_password = self.lineEdit_4.text()

            if password == confirm_password:
                self.textBrowser.setText(f"Hesap kullanıcı adı:{username} ve şifre:{password} ile başarıyla oluşturuldu.")
                with open('Logins.csv', 'a') as csvfile:
                    fieldnames = ['username', 'password', 'module_name']
                    writer = DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'username': username, 'password': password, 'module_name': module_number})

                login = Login()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex() + 1)
            else:
                self.textBrowser.setText("Şifre eşleşmiyor.")
        else:
            self.textBrowser.setText("Modül no yanlış. Tekrar deneyin.")


class AdminRecognition(QDialog):
    def __init__(self):
        super(AdminRecognition, self).__init__()
        loadUi("Admin_log.ui", self)
        self.pushButton.clicked.connect(self.rundatasetdetector)
        self.lineEdit.setPlaceholderText("User Id")
        self.pushButton_2.clicked.connect(self.runtraining)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(len(os.listdir("dataset")))
        self.progressBar.setValue(0)
        self.counter = 0
        self.pushButton_3.clicked.connect(self.back)

    @staticmethod
    def back():
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def rundatasetdetector(self):
        id = self.lineEdit.text()
        if id != "" and id != "1" and id != "2":
            cam = cv2.VideoCapture(0)

            samplenum = 0
            while True:
                _, frame = cam.read()
                samplenum = samplenum + 1
                cv2.imwrite("dataset/User." + str(id) + "." + str(samplenum) + ".jpg", frame[:, :])
                cv2.waitKey(100)
                pass
                cv2.imshow("Faces", frame)
                cv2.waitKey(1)
                self.textBrowser_2.setText("Veriseti oluşturuluyor.")
                if samplenum > 149:
                    break
                pass
            cam.release()
            cv2.destroyAllWindows()
            self.textBrowser_2.setText("Veriseti oluşturuldu.")
            self.checkBox.setChecked(True)
        else:
            self.textBrowser_2.setText("Geçerli Id giriniz.")

    def runtraining(self):
        if self.checkBox.isChecked():

            imagepaths = list(paths.list_images("dataset"))
            knownencodings = []
            ids = []

            for (i, imagePath) in enumerate(imagepaths):
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_locations(rgb, number_of_times_to_upsample=2, model='hog')
                encodings = face_encodings(rgb, boxes, num_jitters=10)
                self.counter = self.counter + 1

                self.progressBar.setValue(self.counter)
                self.textBrowser_2.setText("Eğitim devam ediyor.")
                for encoding in encodings:
                    knownencodings.append(encoding)
                    ids.append(ID)
            data = {"encodings": knownencodings, "names": ids}
            f = open("xx", "wb")
            f.write(dumps(data))
            f.close()
            self.textBrowser_2.setText("Eğitim tamamlandı.")

        else:
            self.textBrowser_2.setText("İlk olarak dataset oluşturun.")


class Recognition(QDialog):
    def __init__(self):
        super(Recognition, self).__init__()
        loadUi("3.ui", self)
        self.pushButton.clicked.connect(self.runrecognition)
        self.control = 0

    def runrecognition(self):
        cap = cv2.VideoCapture(0)
        hog_face_detector = get_frontal_face_detector()
        data = loads(open('face_rec', "rb").read())

        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = hog_face_detector(gray)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_encodings(rgb)
            names = []
            for encoding in encodings:
                matches = compare_faces(data["encodings"], encoding)
                name = "Bilinmeyen Yüz"

                if True in matches:
                    matchedidxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedidxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
                for (face, name) in zip(faces, names):

                    if name == 1 or name == 2:
                        self.textBrowser.setText(f"Kullanıcı{name}")
                        self.checkBox.setChecked(True)
                        sleep(2)
                        self.control = 1

                    else:
                        self.textBrowser.setText(f"{name}")

            cv2.imshow("Frame", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            if self.control == 1:
                ui = Fatigue()
                widget.addWidget(ui)
                widget.setCurrentIndex(widget.currentIndex() + 1)
                break
        cap.release()
        cv2.destroyAllWindows()


class Fatigue(QDialog):
    def __init__(self):
        super(Fatigue, self).__init__()
        loadUi("4.ui", self)
        self.pushButton.clicked.connect(self.runfunction)
        self.array = []
        self.class_names = ['kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirma', 'dogal']
        self.count = 0
        self.a = 0
        self.b = 0

    def runfunction(self):
        model_best = load_model("VGG16-AUX-BEST-70.2.h5")
        hog_face_detector = get_frontal_face_detector()
        dlib_facelandmark = shape_predictor("shape_predictor_68_face_landmarks.dat")
        cap = cv2.VideoCapture(0)

        self.a = time()
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
                roi = frame[y1:y2, x1:x2]
                resized = cv2.resize(roi, (197, 197))

                test_data = np.expand_dims(resized, axis=0)
                test_data = np.vstack([test_data])
                results = model_best.predict(test_data, batch_size=1)
                self.textBrowser_2.setText(self.class_names[np.argmax(results)])
                if self.class_names[np.argmax(results)]=="kizgin":
                    #arduino.write(b'S')
                    pass

                face_landmarks = dlib_facelandmark(gray, face)
                lefteye = []
                righteye = []
                lip = []

                for n in range(36, 42):  # Left Eye
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    lefteye.append((x, y))
                    next_point = n + 1
                    if n == 41:
                        next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                for n in range(42, 48):  # Right Eye
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    righteye.append((x, y))
                    next_point = n + 1
                    if n == 47:
                        next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                for n in range(48, 60):  # lip
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    lip.append((x, y))
                    next_point = n + 1
                    if n == 59:
                        next_point = 48
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

                lip_ratio = self.calc_lip(lip)
                self.lcdNumber_2.display(lip_ratio*100)
                self.yawndetect(lip_ratio*100)
                left_eye = self.calculate_eye(lefteye)
                right_eye = self.calculate_eye(righteye)
                eye = self.calculate_last_eye(left_eye, right_eye)
                opened, closed = self.eye_status_calculation(eye, self.array)
                self.perclos_calc(opened, closed)

            cv2.imshow("image", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def yawndetect(self, lip):
        if lip > 60:
            self.count = self.count + 1
            print(self.count)
        if 29.2 <= self.b-self.a <= 30.8:
            if self.count > 30:
                self.textBrowser_4.setText("Yorgun")
                #arduino.write(b'H')

                self.a = self.b
                time.sleep(0.05)
                self.count = 0

            else:
                self.textBrowser_4.setText("Normal")
                self.a = self.b
                self.count = 0
        self.b = time()

    @staticmethod
    def calculate_eye(eye):
        a = distance.euclidean(eye[1], eye[5])
        b = distance.euclidean(eye[2], eye[4])
        c = distance.euclidean(eye[0], eye[3])
        eye_aspect_ratio = (a + b) / (2.0 * c)
        return eye_aspect_ratio

    @staticmethod
    def calc_lip(lip):
        a = distance.euclidean(lip[2], lip[10])
        b = distance.euclidean(lip[4], lip[8])
        c = distance.euclidean(lip[0], lip[6])
        lip_ratio = (a + b) / (2.0 * c)
        return lip_ratio

    @staticmethod
    def calculate_last_eye(left_eye, right_eye):
        eye = (left_eye + right_eye) / 2
        eye = round(eye, 2)
        return eye

    @staticmethod
    def eye_status_calculation(eye, array):
        if eye < 0.24:
            array.append(0)
        # print(f"eye:{EYE}")
        if eye >= 0.24:
            array.append(1)

        opened = array[-150:].count(1)
        closed = array[-150:].count(0)
        return opened, closed

    def perclos_calc(self, opened, closed):
        total_frame = opened + closed
        perclos = closed * 100 / total_frame
        self.lcdNumber.display(perclos)
        if perclos < 60:
            self.textBrowser_3.setText("Normal")
            #arduino.write(b'L')
        elif perclos >= 60:
            self.textBrowser_3.setText("Yorgun")
            #arduino.write(b'H')



app = QApplication(argv)
mainwindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(650)
widget.setFixedHeight(500)
widget.show()
app.exec_()
