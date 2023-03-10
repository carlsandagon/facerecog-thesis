import cv2 as cv
import os


faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

names = []
for users in os.listdir("dataset"):
    names.append(users)

#img = cv.imread("dataset/carl/1_1.jpg")
vid = cv.VideoCapture(0)

while True:
    _, img = vid.read()

    gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gr, scaleFactor = 1.2, minNeighbors = 5, minSize = (50, 50))

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gr[y:y+h, x:x+w])

        if (confidence < 100):
            id = names[id-1]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv.putText(img, str(id), (x + 5, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(img, str(confidence), (x + 5, y + h - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        #print(id)
        #txt = "Unknown"
        #if not id:
           # cv.putText(img, txt, (x, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv.LINE_AA)
       # else:
            #cv.putText(img, names[id - 1], (x, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv.LINE_AA)

    cv.imshow("Recognize", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()