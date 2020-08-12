from cv2 import cv2
import numpy as np

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("gray_cover.jpg")
roi = img[252: 395, 354: 455]
x = 304
y = 252
width = 455 - x
height = 345 - y
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cap = cv2.VideoCapture(0)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        
def draw_boundary(img,classifier,scaleFactor,mineighbors,color,text):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,mineighbors)
        coords=[]
        for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                coords=[x,y,w,h]
                
        return img,coords

def detect(img,faceCascade):
        img,coords=draw_boundary(img,faceCascade,1.1,10,(68,91,65),'Face')
        
        
        return img


cap = cv2.VideoCapture(0)
while(True):
        ret,frame = cap.read()
        frame=detect(frame,faceCascade)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(mask, (x, y, width, height), term_criteria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        cv2.imshow("mask", mask)
        cv2.imshow('frame',frame)
        
        
        if(cv2.waitKey(1) & 0xFF== ord('q')):
                break
cap.release()
cv2.destroyAllWindows()
