#optical flow and image stabilization test

# imports:
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
#import skimage.io
import cv2
import cv2.cv as cv
from savitzky_golay_filter import savgol
import sys
import copy

# globals:

input_file = "singleRider.avi"
output_file = "out2.avi"
SPEED_UP = 1
CORNER_HARRIS_K = 0.04 
CORNER_HARRIS_K_SIZE = 3
CORNER_HARRIS_BLOCK_SIZE = 2
CORNER_THRESH = 0.05
LK_WINDOW_SIZE = (30, 30)
LK_MAX_LEVEL = 4
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
GF_MAX_CORNERS = 10
GF_QUALITY_LEVEL = 0.3
GF_MIN_DISTANCE = 7
GF_BLOCK_SIZE = 7
FOURCC = cv.CV_FOURCC(*'XVID')
SG_WINDOW = 359
SG_P_ORDER = 3
CROP_THRESH = 0.215


def write_progress(p):
	"""writes to the console the progress that something is at"""
	sys.stdout.write("\r")
	sys.stdout.write(str(p))
	sys.stdout.flush()
   

video = cv2.VideoCapture(input_file)



width = int(video.get(cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv.CV_CAP_PROP_FPS)
total_frames = int(video.get(cv.CV_CAP_PROP_FRAME_COUNT))

x = 0
y = 0

pos = []
pos.append([x, y])

print "computing motion between frames..."

returnVal, img1 = video.read()
cv2.imshow('frame1',img1)
for i in range(1, total_frames):
    
    returnVal, img2 = video.read()
    cv2.imshow('frame2',img2)
    
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    features1 = cv2.goodFeaturesToTrack(gray1, GF_MAX_CORNERS, \
    GF_QUALITY_LEVEL, GF_MIN_DISTANCE, GF_BLOCK_SIZE)
    
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    features2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, features1, \
    nextPts=None, winSize=LK_WINDOW_SIZE, maxLevel=LK_MAX_LEVEL, \
    criteria=LK_CRITERIA)
    color = np.random.randint(0,255,(100,3))
    good_features2 = features2[st==1]
    good_features1 = features1[st==1]
    
    mask = np.zeros_like(img1)
    frame = img2
    
    for t,(new,old) in enumerate(zip(good_features2,good_features1)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[t].tolist(), 2)
        cv2.circle(frame,(a,b),5,color[t].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    
    
    
    diff = good_features1 - good_features2
    #print diff
    
    # if no correspondences are found:
    if len(diff) == 0: 
        diff = np.array([0.0, 0.0])
    
    diff = np.mean(diff, axis=0, dtype=np.float32)
    #cv2.imshow('frame1',frame1)
    
    #print "vec"
    #print vec
    vec = diff
    x += vec[0]
    y += vec[1]
    
    pos.append([x, y])
    
    img1 = img2

    write_progress(str(round(i / float(total_frames), 3) * 100) + \
			"% complete")

pos = np.array(pos)

video.release()

print "translating frames..."
trans = pos

i = 0

video = cv2.VideoCapture(input_file)

x_max, y_max = trans.max(axis = 0)
x_min, y_min = trans.min(axis = 0)
x_crop = max(abs(x_max), abs(x_min))
y_crop = max(abs(y_max), abs(y_min))

print "x_crop =", x_crop, "(", str(round(x_crop * 100. / width, 3)), 
print "% of width),", "y_crop =", y_crop, "(", 
print str(round(y_crop * 100. / height, 3)), "% of height)" 

if x_crop / width > CROP_THRESH or y_crop / height > CROP_THRESH:
    print "Too much video to crop, doing frame overlay instead"

    writer = cv2.VideoWriter(output_file, FOURCC, fps, (width, height))
    returnVal, frame = video.read()
    canvas =  np.zeros((height, width, 3), 'uint8')
    
    for i in range(len(trans)):
        m = np.float32([[1, 0, trans[i][0]], [0, 1, trans[i][1]]])
        dst = cv2.warpAffine(frame, m, (width, height))
    
        matte = (np.clip(dst, 0, 1) - 1) * (-1)
        canvas = canvas * matte + dst
    
        writer.write(canvas.astype(np.uint8))
        write_progress(str(round(i / float(len(trans)), 3) * 100) + \
    				"% complete")
    
        for j in range(SPEED_UP):
	   returnVal, frame = video.read()
           if not returnVal:
   	        break

else:
    print "Cropping frame:",
    print "(" + str(width) + ", "  + str(height) + ") ->",
    width = int(width - (2 * x_crop))
    height = int(height - (2 * y_crop))
    print "(" + str(width) + ", "  + str(height) + ")"
    
    writer = cv2.VideoWriter(output_file, FOURCC, fps, (width, height))
    returnVal, frame = video.read()
    
    for i in range(len(trans)):
        m = np.float32([[1, 0, trans[i][0] - x_crop], 
    					   [0, 1, trans[i][1] - y_crop]])
        dst = cv2.warpAffine(frame, m, (width, height))
    
        writer.write(dst)
        write_progress(str(round(i / float(len(trans)), 3) * 100) + \
    				"% complete")
    
        for j in range(SPEED_UP):
    				returnVal, frame = video.read()
    				if not returnVal:
    				    break
    
print "\n...done!"

video.release()
writer.release()
cv2.destroyAllWindows()

