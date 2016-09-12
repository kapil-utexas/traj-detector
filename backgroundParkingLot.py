#Machine vision code to detect bikes and pedestrians from video taken by a UAV

# Import different modules to be used in the code
import cv2
import numpy as np;
import sys
import time 
import datetime
from sys import exit

#Define a background subtraction algorithm to use from OpenCV
fgbg = cv2.BackgroundSubtractorMOG()
kernel = np.ones((3,3),np.uint8)*1  #Define a kernel for first erosion and dialation
#kernel2 = np.ones((2,2),np.uint8)*1 #Define a kernel for second erosion and dialation  #This is the original kernel 2
kernel2 = np.ones((1,1),np.uint8)*1 #Define a kernel for second erosion and dialation
kernel3 = np.ones((2,2),np.uint8)*1 #Define a kernel for second erosion and dialation

#Import the video into the "cap" variable
#cap = cv2.VideoCapture('carDetectShort.avi')
cap = cv2.VideoCapture('parkingLot.avi')

#Set up parameters for the blob detector
params = cv2.SimpleBlobDetector_Params()
# Change thresholds 
params.minThreshold = 40
params.maxThreshold = 2000  
          
# Filter by Area.
params.filterByArea = True
params.minArea = 200
        
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
    
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
        
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.15

params.minDistBetweenBlobs = 10

    
#Define detector for blob detection
detector = cv2.SimpleBlobDetector(params)


#background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
#Now that the background has been established, move to detect cars in subsequent frames
d=0
while d < 240:
    d=d+1
    ret, im = cap.read()

    backMask = fgbg.apply(im)       #Apply background subtractor to first frame    
    cv2.imshow('mask',backMask)
    erodedTest = cv2.erode(backMask,kernel2,iterations = 2)   #Erode the blobs in the background
    erodedTest = cv2.erode(erodedTest,kernel3,iterations = 1)   #Erode the blobs in the background
    erodedTest = cv2.erode(erodedTest,kernel2,iterations = 1)   #Erode the blobs in the background
    erodedTest = cv2.erode(erodedTest,kernel,iterations = 0)   #Erode the blobs in the background
    cv2.imshow('erodedTest',erodedTest)

    dilatedTest = cv2.dilate(erodedTest,kernel,iterations = 0)   #Erode the blobs in the background
    cv2.imshow('dilatedTest',dilatedTest)


    negative = 255 - dilatedTest
    negative = cv2.erode(negative,kernel,iterations = 2)

    erodedTest = 255 - erodedTest

    cv2.imshow('backMask',negative)
    cv2.imwrite(''.join(['mask',str(d),'.jpg']),backMask)

    cv2.imwrite(''.join(['orig',str(d),'.jpg']),im)
#    keypoints2 = detector.detect(negative)
 #   prevKeyArray = keypoints2
    #Set counter to 0
    '''
    i = 0 
#Set height and width of rectangle
    w = 40
    h = 20
    prevKeyCounter = 0
#Iterate through all detected blobs to draw rectangles
    for s in keypoints2:
        i = i + 1 #Increment counter
            
        x = keypoints2[i-1].pt[0] #Get x-coordinate of center of i-th blob
        y = keypoints2[i-1].pt[1] #Get y-coordinate of center of i-th blob
        
        #Define the x- and y- coordinates of the top left corner of the blob and the bottom right corner
        topx = int(x-w/2)
        topy = int(y+h/2)
        botx = int(x+w/2)
        boty = int(y-h/2)
        
    #Use two corners to draw a rectangle around the center of the i-th blob
        
            #Go through the entire previous list of keypoints to see if there's one nearby the current keypoints
        if d > 10: #Do this only after the first 10 frames are processed

            cv2.rectangle(im,(topx,topy),(botx,boty),(255,0,0),3)
            #Save previous key points in keypoints array


    cv2.imshow('detected',im)
    
   # cv2.imwrite(''.join(['detected',str(d),'.jpg']),im)
   # cv2.imwrite(''.join(['mask',str(d),'.jpg']),backMask)
    '''
