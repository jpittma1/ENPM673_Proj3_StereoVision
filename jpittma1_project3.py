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

# path1 = './curule/'
# path2 = './octagon/'
# path3 = './pentagon/'

# print("Converting provided images into a video...")
# night_drive_video=convertImagesToMovie(path)

# thresHold=180
# start=1 #start video on frame 1
# vid=cv2.VideoCapture('night_drive.avi')
# vid.set(1,start)
# size = vid.shape

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps_out = 1
# videoname_1=('jpittma1_proj3')
# out1 = cv2.VideoWriter(str(videoname_1)+".avi", fourcc, fps_out, size)

dataset_number = 1  #curule
# dataset_number = 2  #octagon
# dataset_number = 3  #pendulum

#K matrixes and baselines from calib.txt files
if dataset_number == 1: #curule
    print("Using 'curule' dataset...")
    K1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline=88.39
    f = K1[0,0]
    folder_name = './curule/'

elif dataset_number == 2:   #octagon
    print("Using 'octagon' dataset...")
    K1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    K2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline=221.76
    f = K1[0,0]
    folder_name = './octagon/'

elif dataset_number == 3:   #pendulum
    print("Using 'pendulum' dataset...")
    K1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    K2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline=537.75
    f = K1[0,0]
    folder_name = './pendulum/'


'''STEP 1: Calibration'''
###---Read in images from appropriate Dataset----####
print("Reading images from dataset...")
images = readImageSet(folder_name, 2)

###---Use SIFT for feature matching---###
sift = cv2.xfeatures2d.SIFT_create()
image0 = images[0].copy()
image1 = images[1].copy()

###---Convert Images to Grayscale
image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

print("Finding matches between images using SIFT...")
kp1, des1 = sift.detectAndCompute(image0_gray, None)
kp2, des2 = sift.detectAndCompute(image1_gray, None)
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x :x.distance)
chosen_matches = matches[0:100]

###---Matching points between images---###
matched_pairs = matchingPairs(chosen_matches, kp1, kp2)

###---Make Images showing SIFT matches connecting images--_###
if dataset_number==1:
    showMatchesOnImages(image0, image1, matched_pairs, [0,0,255], "SIFT_matches_curule.jpg")
elif dataset_number==2:
    showMatchesOnImages(image0, image1, matched_pairs, [0,0,255], "SIFT_matches_octagon.jpg")
elif dataset_number==3:
    showMatchesOnImages(image0, image1, matched_pairs, [0,0,255], "SIFT_matches_pendulum.jpg")

print("Estimating Fundamental Matrix using RANSAC...")
Fundamental_Matrix, matched_pairs_inliers = getInliers(matched_pairs)

print("Fundamental Matrix is: ", Fundamental_Matrix)

print("Estimating Essential Matrix...")
Essential_Matrix = solveEssentialMatrix(K1, K2, Fundamental_Matrix)

print("Essential Matrix is: ", Essential_Matrix)

print("Decomposing Essential 'E' matrix into Translation 'T' and Rotation 'R'...")
##Solve for the solutions of R and T
Rotation, Translation = ExtractCameraPose(Essential_Matrix)
# print("Rotation and Translation as ", Rotation, Translation)


#Get the 3D points of the 4 solutions
pts3D_4 = get3DPoints(K1, K2, matched_pairs_inliers, Rotation, Translation)

count1 = []
count2 = []

rot = np.identity(3)
tran = np.zeros((3,1))
for i in range(len(pts3D_4)):
    pts3D = pts3D_4[i]
    pts3D = pts3D/pts3D[3, :]
    # x = pts3D[0,:]
    # y = pts3D[1, :]
    # z = pts3D[2, :]    

    ##Make sure the Z values are positive
    count2.append(getPositiveCount(pts3D, Rotation[i], Translation[i]))
    count1.append(getPositiveCount(pts3D, rot, tran))

count1 = np.array(count1)
count2 = np.array(count2)
# print("count1 is ", count1)
# print("count2 is ", count2)

count_thresh = int(pts3D_4[0].shape[1] / 2)
# print("count_thresh is ", count_thresh)

idx = np.intersect1d(np.where(count1 > count_thresh), np.where(count2 > count_thresh))

# print("idx is ", idx)
# print("idx is ", idx[0])

rot_best = Rotation[idx[0]]
trans_best = Translation[idx[0]]
X = pts3D_4[idx[0]]
X = X/X[3,:]

print("Estimated Rotation and Translation as ", rot_best)
print("Estimated translation: ",trans_best)

cv2.destroyAllWindows()

'''STEP 2: Rectification'''
###--Solve for Homography Matrixes---###
set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]

##--Draw Unrectified Epipolar Lines on Images--###
# lines1, lines2 = getEpipolarLines(set1, set2, Fundamental_Matrix, image0, image1, "epipolarLines_unrectified.jpg", False)


h1, w1 = image0.shape[:2]
h2, w2 = image1.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), F_best, imgSize=(w1, h1))
print("Estimated H1 and H2 as", H1, H2)


###--Warp and Transform Images---
img1_rectified = cv2.warpPerspective(image0, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(image1, H2, (w2, h2))

set1_rectified = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
set2_rectified = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)

###--Solve for Rectified Fundamental Matrix---###
img1_rectified_draw = img1_rectified.copy()
img2_rectified_draw = img2_rectified.copy()

H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
F_rectified = np.dot(H2_T_inv, np.dot(F_best, H1_inv))


##---Draw Rectified Epipolar lines---###
lines1_rectified, lines2_recrified = getEpipolarLines(set1_rectified, set2_rectified, F_rectified, img1_rectified, img2_rectified, "../Results/RectifiedEpilines_" + str(dataset_number)+ ".png", True)


###---Make Images showing SIFT matches connecting images--_###
if dataset_number==1:
    showMatchesOnImages(image0, image1, matched_pairs, [255,0,0], "curule_rectified.jpg")
elif dataset_number==2:
    showMatchesOnImages(image0, image1, matched_pairs, [255,0,0], "octagon_rectified.jpg")
elif dataset_number==3:
    showMatchesOnImages(image0, image1, matched_pairs, [255,0,0], "pendulum_rectified.jpg")

cv2.destroyAllWindows()
'''STEP 3: Compute Depth Image'''

cv2.destroyAllWindows()