import cv2
import numpy as np
import sys

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0);
    gray = cv2.medianBlur(gray,5)
    ret, threshed = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    mask =  np.zeros(frame.shape[:2],np.uint8)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    if circles is not None:
        for i in circles[0,:]:
            i[2]=i[2]+4
            # Draw on mask
            cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
        dst = cv2.bitwise_and(frame, frame, mask=mask)
        _,threshed = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
        cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea)
        H,W = frame.shape[:2]
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
                break
        crop = dst[y:y+h,x:x+w] 

        ## Display it
        cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        cv2.imwrite("dst.png", dst)
        cv2.imshow("dst.png", dst)
        cv2.waitKey()

cap.release()
cv2.destroyAllWindows()