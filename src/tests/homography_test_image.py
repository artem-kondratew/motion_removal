import cv2 as cv
import numpy as np

#explicit is better than implicit cv2.IMREAD_GRAYSCALE is better than 0
f1 = "img1.png"
f2 = "img1.png"
img1 = cv.imread(f1, cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread(f2, cv.IMREAD_GRAYSCALE)  # trainImage
#CV doesn't hold hands, do the checks.
if (img1 is None) or (img2 is None):
    raise IOError(f"No files {f1} and {f2} found")

# Initiate SIFT detector
sift = cv.SIFT_create() 

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
matches = np.asarray(matches)

print(matches.shape)

if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv.findHomography(src, dst, cv.LMEDS, 5.0)
    print(H)
else:
    raise AssertionError("Can't find enough keypoints")
