import cv2
cam = cv2.VideoCapture(0)
id = input('Enter ID : ')
sampleNum = 0

while True:
    _,frame = cam.read()
    sampleNum = sampleNum + 1
    cv2.imwrite("deneme/User." + str(id) + "." + str(sampleNum) + ".jpg", frame[:, :])
    cv2.waitKey(100)
    pass
    cv2.imshow("Faces", frame)
    cv2.waitKey(1)
    if sampleNum>150:
        break
    pass
cam.release()
cv2.destroyAllWindows()