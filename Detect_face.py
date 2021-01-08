import cv2

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam=cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    is_captured,frame=cam.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinated=face_classifier.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in face_coordinated:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('cam',frame)
    if cv2.waitKey(1)==13:

        cam.release()
        break

cv2.destroyAllWindows()




