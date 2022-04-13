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
import timeit

def readImageSet(folder_name, n_images):
    print("Reading images from ", folder_name)
    images = []
    for n in range(0, n_images):
        image_path = folder_name + "/" + "im" + str(n) + ".png"
        image = cv2.imread(image_path)
        
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def matchingPairs(sift_matches, kp1, kp2):
    matched_pairs = []
    for i, m1 in enumerate(sift_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    
    return matched_pairs

def normalize(uv):
    
    uv_dash = np.mean(uv, axis=0)
    u_dash ,v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:,0] - u_dash
    v_cap = uv[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T

###---Using SVD for solving for Fundamental Matrix---###
def estimateFundamentalMatrix(feature_matches):
    normalized = True

    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]

    if x1.shape[0] > 7:
        if normalized == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        ###---Using SVD for solving for Fundamental Matrix---###
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalized:
            F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

def errorOfFundamentalMatrix(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1_tmp=np.array([x1[0], x1[1], 1]).T
    x2_tmp=np.array([x2[0], x2[1], 1])

    error = np.dot(x1_tmp, np.dot(F, x2_tmp))
    
    return np.abs(error)

def getInliers(features):
    n_iterations = 1000
    error_thresh = 0.02
    inliers_thresh = 0
    chosen_indices = []
    F_matrix = 0

    for i in range(0, n_iterations):
        indices = []
        #select 8 points randomly
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        f_8 = estimateFundamentalMatrix(features_8)
        for j in range(n_rows):
            feature = features[j]
            error = errorOfFundamentalMatrix(feature, f_8)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            F_matrix = f_8

    filtered_features = features[chosen_indices, :]
    
    return F_matrix, filtered_features

###---Using SVD and calibration parameters---###
def solveEssentialMatrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E_corrected = np.dot(U,np.dot(np.diag(s),V))
    return E_corrected

def matchImageSizes(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)

    return images_resized

def showMatchesOnImages(img_1, img_2, matched_pairs, color, file_name):
    image_1 = img_1.copy()
    image_2 = img_2.copy()

    image_1, image_2 = matchImageSizes([image_1, image_2])
    concat = np.concatenate((image_1, image_2), axis = 1)

    if matched_pairs is not None:
        corners_1_x = matched_pairs[:,0].copy().astype(int)
        corners_1_y = matched_pairs[:,1].copy().astype(int)
        corners_2_x = matched_pairs[:,2].copy().astype(int)
        corners_2_y = matched_pairs[:,3].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 2)
    
    # cv2.imshow(file_name, concat)
    cv2.imwrite(file_name, concat)
    # cv2.waitKey() 
    cv2.destroyAllWindows()
    
def ExtractCameraPose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    Rot = []
    Trans = []
    Rot.append(np.dot(U, np.dot(W, V_T)))
    Rot.append(np.dot(U, np.dot(W, V_T)))
    Rot.append(np.dot(U, np.dot(W.T, V_T)))
    Rot.append(np.dot(U, np.dot(W.T, V_T)))
    Trans.append(U[:, 2])
    Trans.append(-U[:, 2])
    Trans.append(U[:, 2])
    Trans.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(Rot[i]) < 0):
            Rot[i] = -Rot[i]
            Trans[i] = -Trans[i]

    return Rot, Trans

##Make sure the Z values are positive
def getPositiveCount(pts3D, R, C):
    I = np.identity(3)
    P = np.dot(R, np.hstack((I, -C.reshape(3,1))))
    P = np.vstack((P, np.array([0,0,0,1]).reshape(1,4)))
    n_positive = 0
    
    for i in range(pts3D.shape[1]):
        X = pts3D[:,i]
        X = X.reshape(4,1)
        Xc = np.dot(P, X)
        Xc = Xc / Xc[3]
        z = Xc[2]
        if z > 0:
            n_positive += 1

    return n_positive

def get3DPoints(K1, K2, inliers, rot_mat, trans_mat):
    pts3D_4 = []
    Rot_1 = np.identity(3)
    Trans_1 = np.zeros((3,1))
    I = np.identity(3)
    P1 = np.dot(K1, np.dot(Rot_1, np.hstack((I, -Trans_1.reshape(3,1)))))

    for i in range(len(trans_mat)):
        x1 = inliers[:,0:2].T
        x2 = inliers[:,2:4].T

        P2 = np.dot(K2, np.dot(rot_mat[i], np.hstack((I, -trans_mat[i].reshape(3,1)))))

        X = cv2.triangulatePoints(P1, P2, x1, x2)  
        pts3D_4.append(X)
        
    return pts3D_4

#solve for X using equation of a line
def getX(line, y):
    x = -(line[1]*y + line[2])/line[0]
    return x

def getEpipolarLines(set1, set2, F, image0, image1, rectified=False):
    lines1, lines2 = [], []
    img_epi1 = image0.copy()
    img_epi2 = image1.copy()

    for i in range(set1.shape[0]):
        x1 = np.array([set1[i,0], set1[i,1], 1]).reshape(3,1)
        x2 = np.array([set2[i,0], set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)

        #solve for in and max x values based on equation of line
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = getX(line2, y2_min)
            x2_max = getX(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = getX(line1, y1_min)
            x1_max = getX(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]

        #draw circles on images for points connecting epipolar lines
        cv2.circle(img_epi2, (int(set2[i,0]),int(set2[i,1])), 10, (0,0,255), -1)
        img_epi2 = cv2.line(img_epi2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (255, 0, int(i*2.55)), 2)
    

        cv2.circle(img_epi1, (int(set1[i,0]),int(set1[i,1])), 10, (0,0,255), -1)
        img_epi1 = cv2.line(img_epi1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (255, 0, int(i*2.55)), 2)

    #Resize images and concatenate back together
    image_1, image_2 = matchImageSizes([img_epi1, img_epi2])
    concat = np.concatenate((image_1, image_2), axis = 1)
    concat = cv2.resize(concat, (1920, 660))

    return lines1, lines2, concat