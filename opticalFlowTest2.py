#Import the necessary toolbox used in this code
import numpy as np
import cv2

cap = cv2.VideoCapture('singleRider.avi')  #Define the movie to be used for bike detection

# params for Shi-Tomasi corner detection
feature_params = dict( maxCorners = 20,
                       qualityLevel = 0.3,
                       minDistance = 57,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


#Tell the video capture which frame I want to grab, when I read from it in the next line.
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,100) 


ret, old_frame = cap.read()                             #Read the first frame from the video 
height , width , layers =  old_frame.shape              #Get the frame height and width
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  #Convert the frame to gray scale
total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))  #Get the total number of frames in the video
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) #Find good features to track in the frame

frame_gray = old_gray


mask = np.zeros_like(old_frame) #Create a mask image for drawing purposes
rows,cols,layers = mask.shape
center = np.array([cols/2, rows/2])
#
#while(1):
n = 0
iKeepList = []
for s in range(0,230):
    n =n + 1
    ret,frame = cap.read()
    if n == 140:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
        iKeepList = []
        iKeepList.append([4])
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    dist = []
    center = rows/2
    maxDist = 0
    h = 20
    w = 40
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        #print i
        a,b = new.ravel()
        c,d = old.ravel()
        #exit()
        cv2.line(mask, (a,b),(c,d), [0,255, 0], 2)
        cv2.circle(frame,(a,b),5,[0,0,255],-1,8)
        x_new = a - cols/2
        y_new = b - rows/2
        x_old = c - cols/2
        y_old = d - rows/2
        distTemp = np.power(np.power(x_new-x_old,2)+np.power(y_new-y_old,2),0.5)

        if distTemp > maxDist and b > 540 and a <500 and a > 250:
            maxDist = distTemp
            xKeep = a
            yKeep = b
            iKeep = i
            
        dist.append([distTemp])
        
       
        #exit()
    iKeepList.append([iKeep])
    count = 0
    for d in np.unique(iKeepList):
        x = good_new[d,0]
        y = good_new[d,1]
        if d != 18:
            cv2.rectangle(frame,(int(x)-w/2,int(y)-h/2),(int(x)+w/2,int(y)+h/2),[0,255,0],2)
    img = cv2.add(frame,mask)
    
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),-theta*1,1)
    #dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('frame',img)
    #cv2.imshow('frame2',dst)
    cv2.imwrite(''.join(['output',str(n),'.jpg']),img)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    


cap.release() #Close the video stream to free memory
