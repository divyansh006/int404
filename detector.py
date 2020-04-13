import cv2
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainningData.yml')
cascadePath =( "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        if (nbr_predicted == 1):
            nbr_predicted = 'Aman'
        elif (nbr_predicted == 2):
            nbr_predicted = 'I dont know'
        cv2.putText(im, str(nbr_predicted) + "--" , (x, y + h), font, 1.1, (0, 255, 0))
        cv2.imshow('face',im)
    k = cv2.waitKey(100)
    if k == 27:
        break
cam.release()
cv2.destroyAllWindows()








