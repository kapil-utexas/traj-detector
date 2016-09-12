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
    
#Define detector for blob detection
detector = cv2.SimpleBlobDetector(params)

g = 0 #Set frame counter for next iteration
d = 0
#Go through each frame and do the detection algorithm

cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,d) #Tell the video capture which frame I want to grab, when I read from it in the next line.
ret, im = cap.read()                        #Read a frame from the video capture
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
avg1 = np.float32(im)
#avg2 = np.float32(im)
#Use 0 to 50 frames with average 1
#Use 150 to 200 with average 1
while(d < 50):
    d= d+ 1
    
    ret, im = cap.read()
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.accumulateWeighted(im,avg1,0.1)
    #cv2.accumulateWeighted(im,avg2,0.01)
     
    res1 = cv2.convertScaleAbs(avg1)
    #res2 = cv2.convertScaleAbs(avg2)
 
    #cv2.imshow('img',im)
    #cv2.imshow('avg1',res1)
    #cv2.imshow('avg2',res2)
    k = cv2.waitKey(20)
 
    if k == 27:
        break
d=150
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,d) #Tell the video capture which frame I want to grab, when I read from it in the next line.
ret, im = cap.read()                        #Read a frame from the video capture
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
avg2 = np.float32(im)

#Use 0 to 50 frames with average 1
#Use 150 to 200 with average 1
while(d < 200):
    d= d+ 1
    
    ret, im = cap.read()
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.accumulateWeighted(im,avg1,0.1)
    cv2.accumulateWeighted(im,avg2,0.1)
     
    #res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)
 
    #cv2.imshow('img',im)
    #cv2.imshow('avg1',res1)
    #cv2.imshow('avg2',res2)
    k = cv2.waitKey(20)
 
    if k == 27:
        break

#cv2.imshow('avg1',res1)
#cv2.imshow('avg2',res2)
#cv2.accumulateWeighted(avg1,avg2,0.5)
#res2 = cv2.convertScaleAbs(avg2)
#cv2.imshow('combined',res2)

#Combining two images
#rows,cols,dpt = avg1.shape
cutoff = 200
#a = avg1[:,10:40,:]
#b = avg2[:,10:40,:]
combined = np.vstack((res1[:cutoff,:],res2[cutoff:,:]))
#combined = np.hstack((res1,res2))
#cv2.imshow('combined2',combined)
#cv2.destroyAllWindows()

background = combined
#background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
#Now that the background has been established, move to detect cars in subsequent frames
d=0
while d < 50:
    d=d+1
    ret, im = cap.read()
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


#    cv2.imshow('original',im)
    subtracted = cv2.subtract(im,background) #Subtarct the previous frame from the current frame 
    subtracted = cv2.convertScaleAbs(subtracted)
    subtracted = np.absolute(subtracted)

    cv2.imshow('subtracted',subtracted)
    #if d < 5:
    #backMask = fgbg.apply(background)       #Apply background subtractor to first frame
    #cv2.imshow('mask',backMask)
    #else:
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
        if (x < 400 or y > 200) and y < 300 and (x - 293 > 2 or x - 293 < -2): #If blob in the right location
            #Go through the entire previous list of keypoints to see if there's one nearby the current keypoints
            if d > 10: #Do this only after the first 10 frames are processed
                for t in range(1,prevKeyCounter): #Go through all the previous keypoints
                    distX = x - prevKeyArray[t-1].pt[0]
                    distY = y - prevKeyArray[t-1].pt[1]
                   # dist = 
                cv2.rectangle(im,(topx,topy),(botx,boty),(255,0,0),3)
                #Save previous key points in keypoints array
                prevKeyCounter = prevKeyCounter + 1
                prevKeyArray[prevKeyCounter - 1] = keypoints2[i-1]

    cv2.imshow('detected',im)
    
    cv2.imwrite(''.join(['detected',str(d),'.jpg']),im)
    cv2.imwrite(''.join(['mask',str(d),'.jpg']),backMask)

exit()
while (g < 30):
    

    g = g + 1 #Increment frame counter
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,g+10) #Tell the video capture which frame I want to grab, when I read from it in the next line.
    ret, im = cap.read()                        #Read a frame from the video capture
    ret, imOrig = cap.read()                       #Read the next frame as well
    
    ##Begin testing Area    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imOrig = cv2.cvtColor(imOrig, cv2.COLOR_BGR2GRAY)

    subtractedTest = cv2.subtract(im,imOrig)
    erodedTest = cv2.erode(subtractedTest,kernel,iterations = 2)   #Erode the blobs in the background

    #END testing area
    fgmask = fgbg.apply(im)       #Apply background subtractor to first frame
    #cv2.imshow(''.join(['true',str(g),'.jpg']),im)
            
                                        
    fgmask = cv2.dilate(fgmask,kernel,iterations = 2)  #Dilate the blobs in the background
    #cv2.imshow('dialated',fgmask)
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)   #Erode the blobs in the background
    #cv2.imshow(''.join(['test',str(g),'.jpg']),fgmask)
    fgmaskKeep = fgmask #Define the background image to keep
    
    
    fgmask = fgbg.apply(imOrig)       #Apply background subtractor
            #cv2.imshow('subtracted background',fgmask)
        
        
    fgmask = cv2.dilate(fgmask,kernel,iterations = 2)  #Dilate the blobs in the background
    #cv2.imshow('dialated',fgmask)
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)   #Erode the blobs in the backgroun
    #cv2.imshow('eroded',fgmask)
    subtracted = cv2.subtract(fgmask,fgmaskKeep) #Subtarct the previous frame from the current frame
    

    
    
    
    subtractedNeg = (255-subtracted)
    subtractedNeg = cv2.erode(subtractedNeg,kernel2,iterations = 4)

    
     
    #Detect blobs using the predefined detector
    keypoints2 = detector.detect(subtractedNeg)
    
    #Set counter to 0
    i = 0 
    #Set height and width of rectangle
    w = 40
    h = 20

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
        cv2.rectangle(imOrig,(topx,topy),(botx,boty),(255,0,0),3)
         

    cv2.imwrite(''.join(['test',str(g),'.jpg']),imOrig)
    if g == 2:
        #cv2.imshow(''.join(['background-mask',str(g),'.jpg']),fgmaskKeep)
        cv2.imshow(''.join(['true',str(g),'.jpg']),im)
        #cv2.imshow(''.join(['current frame mask',str(g),'.jpg']),fgmask)
        cv2.imshow(''.join(['subtracted image',str(g),'.jpg']),subtractedTest)
        cv2.imshow(''.join(['eroded image',str(g),'.jpg']),erodedTest)
#        cv2.imshow(''.join(['subtracted image',str(g),'.jpg']),subtracted)
        #cv2.imshow(''.join(['final detection image',str(g),'.jpg']),imOrig)
        
print datetime.datetime.now()
cap.release()
  