import numpy as np
import cv2
from sys import exit
cap = cv2.VideoCapture('sanJac2.avi')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 250,
                       qualityLevel = 0.2,
                       minDistance = 5 ,
                       blockSize = 2  )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = (0,255, 20)

cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,33) #Tell the video capture which frame I want to grab, when I read from it in the next line.
# Take first frame and find corners in it
ret, old_frame = cap.read()

height , width , layers =  old_frame.shape
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

#video = cv2.VideoWriter("out8.avi",-1,1,(width,height))
total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
rows,cols,layers = mask.shape
center = np.array([cols/2, rows/2])
#
#while(1):
n = 0
for s in range(0,280):
    n =n + 1
    ret,frame = cap.read()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    dist = []
    #cv2.circle(frame,(580,500),15,[0,0,255],-1,8) 
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
        
          
        x_new = a - cols/2
        y_new = b - rows/2
        x_old = c - cols/2
        y_old = d - rows/2
        distTemp = np.power(np.power(x_new-x_old,2)+np.power(y_new-y_old,2),0.5)
        if distTemp > 0.3 and distTemp < 1 and a > 580 and b > 520 and b < 900 and a< 750:
            #
            #cv2.line(mask, (a,b),(c,d), [0,0,255], 2)    
            cv2.rectangle(frame,(int(a)-w/2,int(b)-h/2),(int(a)+w/2,int(b)+h/2),[0,255,0],2)    
            #cv2.circle(frame,(a,b),5,[0,0,255],-1,8)  
        if distTemp > maxDist:
            maxDist = distTemp
            xKeep = a
            yKeep = b
            xKeep_old = c
            yKeep_old = d
            iKeep = i
        dist.append([distTemp])
   
       
        #exit()
    #cv2.line(mask, (xKeep,yKeep),(xKeep_old,yKeep_old), color, 2)
    #cv2.circle(orig,(xKeep,yKeep),5,[0,0,255],-1,8)
    
    img = cv2.add(frame,mask)
    
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),-theta*1,1)
    #dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('frame',img)
    #cv2.imshow('frame2',dst)
    cv2.imwrite(''.join(['output1out',str(n),'.jpg']),img)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    

#cv2.destroyAllWindows()
cap.release()
#video.release()