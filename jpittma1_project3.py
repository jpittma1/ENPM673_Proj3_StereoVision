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

#####-------SELECT DESIRED DATASET-----------###########
dataset_number = 1  #curule
# dataset_number = 2  #octagon
# dataset_number = 3  #pendulum
#######################################################

#K matrixes and baselines from calib.txt files
if dataset_number == 1: #curule
    print("Using 'curule' dataset...")
    K1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline = 88.39  #From calib.txt file
    f = K1[0,0]
    folder_name = './curule/'

elif dataset_number == 2:   #octagon
    print("Using 'octagon' dataset...")
    K1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    K2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline = 221.76 #From calib.txt file
    f = K1[0,0]
    folder_name = './octagon/'

elif dataset_number == 3:   #pendulum
    print("Using 'pendulum' dataset...")
    K1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    K2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline = 537.75 #From calib.txt file
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
# X = pts3D_4[idx[0]]
# X = X/X[3,:]

print("Estimated Rotation: ", rot_best)
print("Estimated translation: ",trans_best)

cv2.destroyAllWindows()
print("Calibration Complete.")
print("######################")

################################################################
'''STEP 2: Rectification'''
print("Rectifying the Epipolar Lines and plotting on Images...")
###--Solve for Homography Matrixes---###
set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]

##--Draw Unrectified Epipolar Lines and Feature Points on Images--###
lines1, lines2, result_unrec = getEpipolarLines(set1, set2, Fundamental_Matrix, image0, image1, False)

###---Make Images showing Rectified Epipolar Lines--_###
if dataset_number==1:
    cv2.imwrite("epipolarLines_crule_unrectified.jpg", result_unrec)
elif dataset_number==2:
    cv2.imwrite("epipolarLines_octagon_unrectified.jpg", result_unrec)
elif dataset_number==3:
    cv2.imwrite("epipolarLines_pendulum_unrectified.jpg", result_unrec)

####----Solve for Homography Matrixes---###
h1, w1 = image0.shape[:2]
h2, w2 = image1.shape[:2]
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), Fundamental_Matrix, imgSize=(w1, h1))
print("Estimated H1 is", H1)
print("Estimated H2 is", H2)

###--Warp and Transform Images---
img1_rect = cv2.warpPerspective(image0, H1, (w1, h1))
img2_rect = cv2.warpPerspective(image1, H2, (w2, h2))

set1_rect = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
set2_rect = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)

###--Solve for Rectified Fundamental Matrix---###
H2_T_inv =  np.linalg.inv(H2.T)
H1_inv = np.linalg.inv(H1)
F_rectified = np.dot(H2_T_inv, np.dot(Fundamental_Matrix, H1_inv))


##---Draw Rectified Epipolar lines and Feature Points---###
lines1_rectified, lines2_recrified, result_rec = getEpipolarLines(set1_rect, set2_rect, F_rectified, img1_rect, img2_rect, True)

###---Make Images showing Rectified Epipolar Lines--_###
if dataset_number==1:
    cv2.imwrite("epipolarLines_crule_rectified.jpg", result_rec)
elif dataset_number==2:
    cv2.imwrite("epipolarLines_octagon_rectified.jpg", result_rec)
elif dataset_number==3:
    cv2.imwrite("epipolarLines_pendulum_rectified.jpg", result_rec)

cv2.destroyAllWindows()
print("Rectification Complete.")
print("######################")

################################################################
'''STEP 3: Correspondance'''
print("Using SSD for calculating the disparity...")

###---Convert to Grayscale--_####
gray1 = cv2.cvtColor(img1_rect,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_rect,cv2.COLOR_BGR2GRAY)

disparities = 15 # window size of image 2
block = 5       # pixel size of block in image 1

height, width = gray1.shape
disparity_img = np.zeros(shape = (height,width))

start = timeit.default_timer()

###---Apply Matching Windows concept with SSD---###
#Computing the Disparity Map by comparing matches along epipolar lines of the two images using SSD
for i in range(block, gray1.shape[0] - block - 1):
    for j in range(block + disparities, gray1.shape[1] - block - 1):
        ssd = np.empty([disparities, 1])
        l = gray1[(i - block):(i + block), (j - block):(j + block)]
        height, width = l.shape
        for d in range(0, disparities):
            r = gray2[(i - block):(i + block), (j - d - block):(j - d + block)]
            ssd[d] = np.sum((l[:,:]-r[:,:])**2)
        disparity_img[i, j] = np.argmin(ssd)

stop = timeit.default_timer()
print("SSD took ", stop-start, " seconds")

print("Disparity (SSD) is: ", ssd)
# print("Disparity_img is: ", disparity_img)

print("SSD Complete. Rescaling to 0-255...")
##---Rescale to 0-255
img_dispair = ((disparity_img/disparity_img.max())*255).astype(np.uint8)
# cv2.imshow('Disparity grayscale',final_img)

print("Creating heatmap image using 'inferno'...")
colormap = plt.get_cmap('inferno')
img_heatmap = (colormap(img_dispair) * 2**16).astype(np.uint16)[:,:,:3]
img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_RGB2BGR)
# cv2.imshow('Disparity heatmap', img_heatmap)

if dataset_number==1:
    cv2.imwrite("Disparity_grayscale_crule.jpg", img_dispair)
    cv2.imwrite("Disparity_heatmap_crule.jpg", img_heatmap)
elif dataset_number==2:
    cv2.imwrite("Disparity_grayscale_octagon.jpg", img_dispair)
    cv2.imwrite("Disparity_heatmap_octagon.jpg", img_heatmap)
elif dataset_number==3:
    cv2.imwrite("Disparity_grayscale_pendulum.jpg", img_dispair)
    cv2.imwrite("Disparity_heatmap_pendulum.jpg", img_heatmap)

cv2.destroyAllWindows()
print("Correspondance Complete.")
print("###########################")

################################################################
'''STEP 4: Compute Depth Image'''
print("Computing Depth of Images...")

depth = np.zeros(shape=gray1.shape).astype(float)
depth[img_dispair > 0] = (f * baseline) / (img_dispair[img_dispair > 0])

# print("Depth is: ", depth)

##---Rescale to 0-255
img_depth = ((depth/depth.max())*255).astype(np.uint8)

print("Creating heatmap image using 'inferno'...")
colormap = plt.get_cmap('inferno')
img_depth_heatmap = (colormap(img_depth) * 2**16).astype(np.uint16)[:,:,:3]
img_depth_heatmap  = cv2.cvtColor(img_depth_heatmap, cv2.COLOR_RGB2BGR)

if dataset_number==1:
    cv2.imwrite("Depth_grayscale_crule.jpg", img_depth)
    cv2.imwrite("Depth_heatmap_crule.jpg", img_depth_heatmap)
elif dataset_number==2:
    cv2.imwrite("Depth_grayscale_octagon.jpg", img_depth)
    cv2.imwrite("Depth_heatmap_octagon.jpg", img_depth_heatmap)
elif dataset_number==3:
    cv2.imwrite("Depth_grayscale_pendulum.jpg", img_depth)
    cv2.imwrite("Depth_heatmap_pendulum.jpg", img_depth_heatmap)

print("Depth Complete. Project 3 Complete.")

cv2.destroyAllWindows()