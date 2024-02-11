import cv2 as cv
import numpy as np
from scipy.stats import kurtosis
from sklearn.cluster import DBSCAN


def motion_comp(prev_frame, curr_frame, num_points=500, points_to_use=500, transform_type='affine'):
    """ Obtains new warped frame1 to account for camera (ego) motion
        Inputs:
            prev_frame - first image frame
            curr_frame - second sequential image frame
            num_points - number of feature points to obtain from the images
            points_to_use - number of point to use for motion translation estimation 
            transform_type - type of transform to use: either 'affine' or 'homography'
        Outputs:
            A - estimated motion translation matrix or homography matrix
            prev_points - feature points obtained on previous image
            curr_points - feature points obtaine on current image
        """
    transform_type = transform_type.lower()
    assert(transform_type in ['affine', 'homography'])

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_RGB2GRAY)
    curr_gray = cv.cvtColor(curr_frame, cv.COLOR_RGB2GRAY)

    # get features for first frame
    corners = cv.goodFeaturesToTrack(prev_gray, num_points, qualityLevel=0.01, minDistance=10)

    # get matching features in next frame with Sparse Optical Flow Estimation
    matched_corners, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)

    # reformat previous and current corner points
    prev_points = corners[status==1]
    curr_points = matched_corners[status==1]

    # sub sample number of points so we don't overfit
    if points_to_use > prev_points.shape[0]:
        points_to_use = prev_points.shape[0]

    index = np.random.choice(prev_points.shape[0], size=points_to_use, replace=False)
    prev_points_used = prev_points[index]
    curr_points_used = curr_points[index]

    # find transformation matrix from frame 1 to frame 2
    if transform_type == 'affine':
        H, _ = cv.estimateAffine2D(prev_points_used, curr_points_used, method=cv.RANSAC)
    elif transform_type == 'homography':
        H, _ = cv.findHomography(prev_points_used, curr_points_used)

    return H, prev_points, curr_points


def main():
    print('opencv', cv.__version__)

    first = True

    cap = cv.VideoCapture('/home/user/ros2_ws/src/tum/dataset/video.mp4')

    while True:
        ret, rgb = cap.read()

        if not ret:
            break

        # cv.imshow('rgb', rgb)
        # cv.waitKey(25)

        if first:
            prev_rgb = rgb
            first = False
            continue

        H, prev_pts, curr_pts = motion_comp(prev_rgb, rgb, num_points=10000, points_to_use=10000, transform_type='homography')
        compensated_pts = np.hstack((prev_pts, np.ones((len(prev_pts), 1)))) @ H.T
        compensated_pts = compensated_pts[:, :2]

        # print(f" Prev Key Points: {np.round(prev_pts[100], 2)} \n",
        #     f"Compensated Key Points: {np.round(compensated_pts[100], 2)} \n",
        #     f"Current Key Points: {np.round(curr_pts[100], 2)}")
        
        compensated_flow = curr_pts - compensated_pts

        x = np.linalg.norm(compensated_flow, ord=2, axis=1) 

        c = 2 # tunable scale factor

        # We expect a Leptokurtic distribution with extrememly long tails
        if kurtosis(x, bias=False) < 1:
                c /= 2 # reduce outlier hyparameter

        # get outlier bound (only care about upper bound since lower values are not likely movers)
        upper_bound = np.mean(x) + c*np.std(x, ddof=1)

        motion_idx = (x >= upper_bound)
        motion_pts = curr_pts[motion_idx]

        # add additional motion data for clustering
        motion = compensated_pts[motion_idx] - curr_pts[motion_idx] 
        magnitude = np.linalg.norm(motion, ord=2, axis=1)
        angle = np.arctan2(motion[:, 0], motion[:, 1]) # horizontal/vertial

        motion_data = np.hstack((motion_pts, np.c_[magnitude], np.c_[angle]))

        cluster_model = DBSCAN(eps=50.0, min_samples=3)
        cluster_model.fit(motion_data)

        angle_thresh = 0.1 #  radians
        edge_thresh = 50   # pixels
        max_cluster_size = 80 # number of cluster points

        clusters = []
        w = rgb.shape[1]
        h = rgb.shape[0]    
        far_edge_array = np.array([w - edge_thresh, h - edge_thresh])
        for lbl in np.unique(cluster_model.labels_):
            cluster_idx = cluster_model.labels_ == lbl

            # get standard deviation of the angle of apparent motion 
            angle_std = angle[cluster_idx].std(ddof=1)
            if angle_std <= angle_thresh:
                cluster = motion_pts[cluster_idx]

                # remove clusters that are too close to the edges and ones that are too large
                centroid = cluster.mean(axis=0)
                if (len(cluster) < max_cluster_size) \
                    and not (np.any(centroid < edge_thresh) or np.any(centroid > far_edge_array)):
                    clusters.append(cluster)

        # draw detected clusters
        for j, cluster in enumerate(clusters):
            color = get_color((j+1)*5)
            frame2 = plot_points(frame2, cluster, radius=10, color=color)


if __name__ == '__main__':
    main()
