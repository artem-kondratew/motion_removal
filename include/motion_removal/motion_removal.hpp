#ifndef MOTION_REMOVAL_MOTION_REMOVAL_HPP
#define MOTION_REMOVAL_MOTION_REMOVAL_HPP


#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/video/tracking.hpp>
#include <opencv4/opencv2/optflow.hpp>

#include <opencv4/opencv2/calib3d.hpp>

#include <opencv4/opencv2/features2d.hpp>


namespace motion_removal {

cv::Mat calcOpticalFlowFurnerback(cv::Mat prev, cv::Mat curr);
cv::Mat calcOpticalFlowSparceToDense(cv::Mat prev, cv::Mat curr);
cv::Mat visualizeOpticalFlow(cv::Mat flow, std::string win_name);

cv::Mat calcHomography(cv::Mat prev, cv::Mat curr, bool use_good_matches);

cv::Mat sparce(cv::Mat prev, cv::Mat curr);
cv::Mat warp(cv::Mat prev, cv::Mat curr);

cv::Mat motionRemoval(cv::Mat curr);

}


#endif // MOTION_REMOVAL_MOTION_REMOVAL_HPP
 