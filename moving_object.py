#import required libraries

import cv2
import imutils
import time

cam=cv2.VideoCapture(0)       # set the camera device 
time.sleep(1)                
firstframe=None               #intialize first frame to none
area=400                   

while True:
    _,img=cam.read()         #read the image from camera
    text="Normal"            #Normal text when no object is detected
    img=imutils.resize(img,width=1000)                   # window width for capturing video =1000 
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        #convert the obtained color image  to gray
    blurr=cv2.GaussianBlur(gray_img,(21,21),0)           # add a gussian blurr to smoothen the image background
    if firstframe is None:
        firstframe=blurr                                 # intialize the first time obtained image as firstframe if above condition executes true 
        continue                                         # execute from while loop again i.e reading 2nd frame image  
    imgDiff=cv2.absdiff(firstframe,gray_img)             #absolute difference function to detect difference in first frame and continously obtaining gray scale image 
    threshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]      #apply threshold to imgDiff
    thresImg=cv2.dilate(threshImg,None,iterations=2)                  #dialte the imgDiff  
    cnts=cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)       #find the contours in diff image
    cnts=imutils.grab_contours(cnts)                                  #connect those contours
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)                           # obtain the respective axis ,height and width from countours 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)          #draw a rectangle from a obtained coordinates with red color and thickness of 2
        text="Moving object Detected"                           #txt msg when object is detected

    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)      #adding text to frame at (10,20)co ordinates with red colour
    cv2.imshow("cameraFeed",img)                                                # show the above all processed image continously in the form of video
    if cv2.waitKey(1) & 0xFF ==ord('q'):                                        #untill user enters q key in keyboard
        break
cam.release()                                                    # then release camera 
cv2.destroyAllWindows()                                          # destroy all windows

    
        
            
    
