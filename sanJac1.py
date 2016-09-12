#Import necessary libraries for this code
import numpy as np
import cv2
from sys import exit

cap = cv2.VideoCapture('sanJac1.avi') #Define video to use

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 300,
                       qualityLevel = 0.2,
                       minDistance = 5 ,
                       blockSize = 2  )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,100) #Tell the video capture which frame I want to grab, when I read from it in the next line.

ret, old_frame = cap.read() #Read one frame from the video

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) #Make that frame gray-scale

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)  #Find the good features in this frame

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#Define the height and width of a rectangle to be drawn later
h = 20
w = 40

n = 0 #Set frame counter to 0

for s in range(0,50):  #Repeat for the next 50 frames

    n =n + 1                 #Increment frame counter
    ret,frame = cap.read()   #Read in the next frame from the video
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #Make the frame gray-scale
    
    # calculate optical flow using the features found in the previous frame
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good features
    good_new = p1[st==1] #Good features in the new frame
    good_old = p0[st==1] #Good features in the old frame
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)): #Iterate through all the features in the new frame
        a,b = new.ravel()  #Get coordinates of features in new frame
        c,d = old.ravel()  #Get coordinates of features in old frame
        
        cv2.circle(frame,(a,b),5,[0,0,255],-1,8) #Draw a red circle centered at the new feature coordinates

        distTemp = np.power(np.power(a-c,2)+np.power(b-d,2),0.5) #Calculate the distance traveled over the past frame by this feature
        
        #If feature traveled less than a max distance and more than a min distance (this filters for speed of features)
            #Additional constraints: If features within a certain region of the image. 
        if distTemp > 0.2 and distTemp < 2 and a > 580 and b > 500 and b < 900 and a< 750: 
        #if distTemp > 1.5 and distTemp < 2:    
            cv2.line(mask, (a,b),(c,d), [0,0,255], 7)   #Draw a line from previous location of the feature to the current location 
            cv2.rectangle(frame,(int(a)-w/2,int(b)-h/2),(int(a)+w/2,int(b)+h/2),[0,255,0],2)   #Draw a rectangle around the feature 
        

      
    img = cv2.add(frame,mask) #Combine the mask that includes the features and the original video frame
    
    cv2.imshow('frame',img)  #Show the combined

    #These lines relate to displaying the image on the screen
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()   #Copy the current frame to old_gray

   
    p0 = good_new.reshape(-1,1,2)  #Set the old features to the current new features before returning to top to get the next frame



cap.release()
