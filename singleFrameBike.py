#testing a single frame to get the bike detection algorithm better and faster than processing the whole thing over and over again
import cv2
import numpy as np;
import sys
import time 
import datetime
from sys import exit

kernel = np.ones((3,3),np.uint8)*1  #Define a kernel for first erosion and dialation
#kernel2 = np.ones((2,2),np.uint8)*1 #Define a kernel for second erosion and dialation  #This is the original kernel 2
kernel2 = np.ones((1,1),np.uint8)*1 #Define a kernel for second erosion and dialation
kernel3 = np.ones((2,2),np.uint8)*1 #Define a kernel for second erosion and dialation

#Import the video into the "cap" variable
#cap = cv2.VideoCapture('carDetectShort.avi')
#cap = cv2.VideoCapture('singleRider.avi')

#Set up parameters for the blob detector
params = cv2.SimpleBlobDetector_Params()
# Change thresholds 
params.minThreshold = 240
params.maxThreshold = 2000  
          
# Filter by Area.
params.filterByArea = True
params.minArea = 1000
        
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.48
    
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.95
        
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.35
    
#Define detector for blob detection
detector = cv2.SimpleBlobDetector(params)

initial = cv2.imread('mask149.jpg')
im = initial

cv2.imshow('starting point', initial)

#cv2.imshow('mask',backMask)
erodedTest = cv2.erode(initial,kernel2,iterations = 2)   #Erode the blobs in the background
erodedTest = cv2.erode(erodedTest,kernel3,iterations = 1)   #Erode the blobs in the background
erodedTest = cv2.erode(erodedTest,kernel2,iterations =0)   #Erode the blobs in the background
erodedTest = cv2.erode(erodedTest,kernel,iterations = 0)   #Erode the blobs in the background
cv2.imshow('erodedTest',erodedTest)

dilatedTest = cv2.dilate(erodedTest,kernel3,iterations = 4)   #Erode the blobs in the background
cv2.imshow('dilatedTest',dilatedTest)


negative = 255 - dilatedTest
negative = cv2.erode(negative,kernel,iterations = 5)

#erodedTest = 255 - erodedTest

cv2.imshow('negative',negative)

keypoints2 = detector.detect(negative)
prevKeyArray = keypoints2
#Set counter to 0
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
    if (x < 600 and y > 550 and x > 250):
        #   #Go through the entire previous list of keypoints to see if there's one nearby the current keypoints
        #  if d > 10: #Do this only after the first 10 frames are processed
        #     for t in range(1,prevKeyCounter): #Go through all the previous keypoints
        #        distX = x - prevKeyArray[t-1].pt[0]
            #       distY = y - prevKeyArray[t-1].pt[1]
                # dist = 
        cv2.rectangle(im,(topx,topy),(botx,boty),(255,0,0),3)
cv2.imshow('final result',im)