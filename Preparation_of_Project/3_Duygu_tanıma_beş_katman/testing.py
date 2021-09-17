import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image


model_best = load_model("face_model_yeni.h5")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("frontalface.xml")
class_names = ['kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirma', 'dogal']
while True:
    _,frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Bulunan yüzleri dikdörtgen içine alıyoruz.
        roi_gray = gray[y:y + w, x:x + h]
        resized = cv2.resize(roi_gray,(48,48))
        test_data = image.img_to_array(resized)
        test_data = np.expand_dims(test_data, axis=0)
        test_data = np.vstack([test_data])

        results = model_best.predict(test_data, batch_size=1)

        print("Sınıflandırma sonucu en yüksek oranla:", class_names[np.argmax(results)])
        cv2.imshow("roi_Gray", roi_gray)


    cv2.imshow("image", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()