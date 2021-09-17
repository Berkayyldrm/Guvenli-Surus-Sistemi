import cv2
import numpy as np
from keras.models import load_model



model_best = load_model("modeller/VGG16-AUX-BEST-70.2.h5")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("frontalface.xml")
class_names = ['kizgin', 'igrenme', 'korku', 'mutlu', 'uzgun', 'sasirma', 'dogal']
while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Bulunan yüzleri dikdörtgen içine alıyoruz.
        roi= frame[y:y + h, x:x + w]
        resized = cv2.resize(roi,(197,197))

        test_data = np.expand_dims(resized, axis=0)
        test_data = np.vstack([test_data])

        results = model_best.predict(test_data, batch_size=1)
        print(results)
        print("Sınıflandırma sonucu en yüksek oranla:", class_names[np.argmax(results)])
        cv2.imshow("roi_Gray", roi)


    cv2.imshow("image", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

