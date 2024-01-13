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

cv::Mat calcOpticalFlowFurnerback(cv::Mat curr_gray, cv::Mat prev_gray);
cv::Mat calcOpticalFlowSparceToDense(cv::Mat curr_gray, cv::Mat prev_gray);
void visualizeOpticalFlow(cv::Mat flow, std::string win_name);

cv::Mat calcHomography(cv::Mat prev, cv::Mat curr);

cv::Mat motionRemoval(cv::Mat curr);

}


#endif // MOTION_REMOVAL_MOTION_REMOVAL_HPP
 