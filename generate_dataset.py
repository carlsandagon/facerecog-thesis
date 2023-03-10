import cv2 as cv

from pathlib import Path

faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

vid = cv.VideoCapture(0)

print("enter the name and id:")
userId = input()
userName = input()
count = 1

def saveImage(img, userName, userId, imgId):
    Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
    cv.imwrite("dataset/{}/{}_{}.jpg".format(userName, userId, imgId), img)


while True:
    _, img = vid.read()

    originalImg = img

    gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gr, scaleFactor = 1.2, minNeighbors = 5, minSize = (50, 50))

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        coords = [x, y, w, h]

    cv.imshow("id", img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        if count <= 5:
            roi_img = originalImg[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
            saveImage(roi_img, userName, userId, count)
            count += 1
        else:
            break
    elif key == ord('q'):
        break


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()