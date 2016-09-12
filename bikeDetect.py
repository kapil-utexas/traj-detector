#Machine vision code to detect bikes and pedestrians from video taken by a UAV

# Standard imports
import cv2
import numpy as np;
import sys
import time 

#Define a background subtraction algorithm to use from OpenCV
fgbg = cv2.BackgroundSubtractorMOG()
kernel = np.ones((3,3),np.uint8)*1  #Define a kernel for erosion and dialation

g = 0
#Go through each image and do the detection algorithm
while (g < 30):
    
    #Import the video into the "cap" variable
    cap = cv2.VideoCapture('carDetectShort.avi')

    d=0 #Reset background counter
    g = g + 1 #Increment frame counter
    
    while(d<200):
        
        ret, im = cap.read()          #Read an image from the video capture
        d=d+1                         #increment counter
        fgmask = fgbg.apply(im)       #Apply background subtractor
        fgmask = cv2.dilate(fgmask,kernel,iterations = 2)  #Dilate the blobs in the background
        fgmask = cv2.erode(fgmask,kernel,iterations = 1)   #Erode the blobs in the backgroun
        

        #cv2.imshow('frame',fgmask1)
        #cv2.imshow('frame2',fgmask2)
        #cv2.imshow('frame3',fgmask)
        #time.sleep(.1)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        if d == 10 + g:
            fgmaskKeep = fgmask
        if d == 11 + g:
            subtracted = cv2.subtract(fgmask,fgmaskKeep)
            imOrig = im
            
            break


            #cv2.imshow('subtracted',subtracted)
            #cv2.imshow('true',imOrig)
        #if d > 10:
        #   cv2.imwrite(''.join(['image',str(d),'.jpg']),fgmask)
        #  cv2.imwrite(''.join(['true',str(d),'.jpg']),im)
    
    #Set the image to the one where the background is substracted(?)
    im = subtracted 

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    kernel = np.ones((2,2),np.uint8)*1
    im = (255-im)
    im2 = cv2.erode(im,kernel,iterations = 2)
    
    fgmask = 255-fgmask #Get the negative of the image
    
    fgmask = cv2.erode(fgmask,kernel,iterations = 2) #Erode the negative

    # Change thresholds 
    params.minThreshold = 40
    params.maxThreshold = 2000
            
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
        
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
        
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    #Define detector for blob detection
    detector = cv2.SimpleBlobDetector(params)
    
    #Save original image in 2 separate images
    subIM1 = imOrig    
    subIM2 = imOrig
    
    #Detect blobs using the predefined detector
    keypoints2 = detector.detect(im2)
    
    #Set counter to 0
    i = 0 
    #Set height and width of rectangle
    w = 40
    h = 20

    #Iterate through all detected blobs
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
        cv2.rectangle(subIM1,(topx,topy),(botx,boty),(255,0,0),3)
        
    #cv2.imshow("Detected Cars", subIM1)
    #cv2.imshow("final image2",im2)
    
    keypoints1 = detector.detect(fgmask)
    i = 0
    for s in keypoints1:
        i = i + 1
        #cv2.circle(subIM2,(int(keypoints1[i-1].pt[0]),int(keypoints1[i-1].pt[1])),20,(0,255,0),3) 
        
    #cv2.imshow("Keypoints-sub1", subIM2)
    #cv2.imshow("final image1",fgmask)
    
    
    cv2.imwrite(''.join(['finalVer2',str(g),'.jpg']),subIM1)
        