#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #3 Functions

import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import histogram_bin_edges, linalg as LA
import matplotlib.pyplot as plt
import sys
import math
import os
from os.path import isfile, join

def convertImagesToMovie(folder):
    fps =   3
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoname=('night_drive')
    frames = []
    
    files = [f for f in os.listdir(folder) if isfile(join(folder, f))]
    
    #pictures were not in order, sadly
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    
    for i in range(len(files)):
        filename=folder + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        # cv2.imshow('image', img)
        # print("size is ", size)
        if img is not None:
            frames.append(img)
        else:
            print("Failed to read")
    
    video = cv2.VideoWriter(str(videoname)+".avi",fourcc, fps, size)
    
    for i in range(len(frames)):
        # writing to a image array
        video.write(frames[i])
    
    #convert heightxwidth from 370,1224 to 480x640so divisible by 8
    cap=cv2.VideoCapture(str(videoname)+".avi")
    size = (640, 480)
    # size = (1280, 720)
    video_new = cv2.VideoWriter(str(videoname)+".avi",fourcc, fps, size)
    
    while True:
        ret,frame_new=cap.read()
        if ret==True:
            b=cv2.resize(frame_new,size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            video_new.write(b)
        else:
            break
    video_new.release()
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    
    return video_new