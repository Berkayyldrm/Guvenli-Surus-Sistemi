import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import csv
import face_recognition
import pickle
import dlib
import cv2
import numpy as np
from keras.models import load_model
import time
from imutils import paths
import os
from skimage import feature as ft
from scipy.spatial import distance



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
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['username'] == username and row["password"] == password:
                    self.textBrowser.setText("Login Successful")
                    recog = Recognition()
                    widget.addWidget(recog)
                    widget.setCurrentIndex(widget.currentIndex() + 1)
                    break

                else:
                    with open('AdminLog.csv', newline='') as csvfile2:
                        reader = csv.DictReader(csvfile2)
                        for row2 in reader:
                            if row2['username'] == username and row2["password"] == password:
                                self.textBrowser.setText("Admin Login Successful")
                                adminrecog = AdminRecognition()
                                widget.addWidget(adminrecog)
                                widget.setCurrentIndex(widget.currentIndex() + 1)
                                break

                            else:
                                self.textBrowser.setText("Username or password is incorrect. Try again.")
                                break
                    self.textBrowser.setText("Username or password is incorrect. Try again.")

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
    def back(self):
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)
    def learn(self):
        module_number = self.lineEdit.text()
        username = self.lineEdit_2.text()
        with open('Logins.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['username'] == username and row["module_number"] == module_number:
                    self.textBrowser.setText(f"Sifreniz = {row['password']}")
                    break
                else:
                    self.textBrowser.setText("Username or password is incorrect. Try again.")


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
                self.textBrowser.setText(f"Successfully created acc with username:{username} and password :{password}")
                with open('Logins.csv', 'a') as csvfile:
                    fieldnames = ['username', 'password', 'module_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'username': username, 'password': password, 'module_name': module_number})

                login = Login()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex() + 1)
            else:
                self.textBrowser.setText("Password does not match")
        else:
            self.textBrowser.setText("Module No is incorrect. Try again")


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

            sampleNum = 0
            while True:
                _, frame = cam.read()
                sampleNum = sampleNum + 1
                cv2.imwrite("dataset/User." + str(id) + "." + str(sampleNum) + ".jpg", frame[:, :])
                cv2.waitKey(100)
                pass
                cv2.imshow("Faces", frame)
                cv2.waitKey(1)
                self.textBrowser_2.setText("Dataset oluşturuluyor.")
                if sampleNum > 149:
                    break
                pass
            cam.release()
            cv2.destroyAllWindows()
            self.textBrowser_2.setText("Dataset oluşturuldu.")
            self.checkBox.setChecked(True)
        else:
            self.textBrowser_2.setText("Geçerli Id giriniz.")


    def runtraining(self):
        if self.checkBox.isChecked() == True:

            imagePaths = list(paths.list_images("dataset"))
            knownEncodings = []
            IDs = []

            for (i, imagePath) in enumerate(imagePaths):
                ID = int(os.path.split(imagePath)[-1].split('.')[1])
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=2, model='hog')
                encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=10)
                self.counter = self.counter + 1

                self.progressBar.setValue(self.counter)
                self.textBrowser_2.setText("Eğitim devam ediyor.")
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    IDs.append(ID)
            data = {"encodings": knownEncodings, "names": IDs}
            f = open("xx", "wb")
            f.write(pickle.dumps(data))
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
        hog_face_detector = dlib.get_frontal_face_detector()
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
                    matchedidxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedidxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
                for (face, name) in zip(faces, names):

                    if name == 1:
                        self.textBrowser.setText(f"Kullanıcı{name}")
                        self.checkBox.setChecked(True)
                        time.sleep(2)
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
        hog_face_detector = dlib.get_frontal_face_detector()
        dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        cap = cv2.VideoCapture(0)
        model = pickle.load(open("model.kayit", "rb"))
        self.a = time.time()
        while True:
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = hog_face_detector(gray)
            featureVector = []
            featureVector2 = []
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

                face_landmarks = dlib_facelandmark(gray, face)
                u1 = face_landmarks.part(36).x
                u2 = face_landmarks.part(39).x
                o1 = face_landmarks.part(37).y
                o2 = face_landmarks.part(40).y

                lefteye = gray[o1 - 15:o2 + 15, u1 - 15:u2 + 15]
                cv2.rectangle(img=frame, pt1=(u1 - 15, o1 - 15), pt2=(u2 + 15, o2 + 15), color=(0, 255, 255),
                              thickness=3)

                resize = cv2.resize(lefteye, (24, 24))
                fd, features = ft.hog(resize, orientations=18, pixels_per_cell=(2, 2),
                                      cells_per_block=(2, 2), block_norm='L1', visualize=True, transform_sqrt=True,
                                      feature_vector=True)
                featureVector.extend(np.array(features).reshape(1, -1))
                b = np.array(model.predict(featureVector))
                self.array.extend(b)



                # sağ göz
                c1 = face_landmarks.part(42).x
                c2 = face_landmarks.part(45).x
                b1 = face_landmarks.part(43).y
                b2 = face_landmarks.part(46).y

                righteye = gray[b1 - 15:b2 + 15, c1 - 15:c2 + 15]
                cv2.rectangle(img=frame, pt1=(c1 - 15, b1 - 15), pt2=(c2 + 15, b2 + 15), color=(0, 255, 255),
                              thickness=3)

                resize2 = cv2.resize(righteye, (24, 24))
                fd, features2 = ft.hog(resize2, orientations=18, pixels_per_cell=(2, 2),
                                       cells_per_block=(2, 2), block_norm='L1', visualize=True, transform_sqrt=True,
                                       feature_vector=True)
                featureVector2.extend(np.array(features2).reshape(1, -1))
                a = np.array(model.predict(featureVector2))
                self.array.extend(a)

                opened, closed = self.eye_status_calculation(self.array)

                self.perclos_calc(opened, closed)

                lip = []

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
                self.lcdNumber_2.display(lip_ratio * 100)
                self.yawndetect(lip_ratio * 100)
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
                self.a = self.b
                self.count = 0

            else:
                self.textBrowser_4.setText("Normal")
                self.a = self.b
                self.count = 0
        self.b = time.time()

    @staticmethod
    def calc_lip(lip):
        a = distance.euclidean(lip[2], lip[10])
        b = distance.euclidean(lip[4], lip[8])
        c = distance.euclidean(lip[0], lip[6])
        lip_ratio = (a + b) / (2.0 * c)
        return lip_ratio

    def eye_status_calculation(self,array):
        opened = array[-150:].count(1)  # son 150 tane frame'i değerlendir(1 olanlanarın sayısını al) (15-10 sn arasındaki frameleri değerlendiriyor fps(pc hızına bağğlı))
        closed = array[-150:].count(0)  # tam tersi 0 ları alıyor
        return opened, closed

    def perclos_calc(self, opened, closed):
        total_frame = opened + closed
        perclos = closed * 100 / total_frame
        self.lcdNumber.display(perclos)
        if perclos < 60:
            self.textBrowser_3.setText("Normal")
        elif perclos >= 60:
            self.textBrowser_3.setText("Yorgun")


app = QApplication(sys.argv)
mainwindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(650)
widget.setFixedHeight(500)
widget.show()
app.exec_()
