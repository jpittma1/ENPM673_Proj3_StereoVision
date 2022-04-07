#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #3

#********************************************
#Requires the following in same folder to run:
# 1) "functions.py"
# 2) folders "curule", "octagon", and "pendulum"
#       with 2 images each plus calib.txt
#********************************************
from functions import *

path1 = './curule/'
path2 = './octagon/'
path3 = './pentagon/'

# print("Converting provided images into a video...")
# night_drive_video=convertImagesToMovie(path)

thresHold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('night_drive.avi')
vid.set(1,start)
size = vid.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps_out = 1
videoname_1=('jpittma1_proj3')
out1 = cv2.VideoWriter(str(videoname_1)+".avi", fourcc, fps_out, size)