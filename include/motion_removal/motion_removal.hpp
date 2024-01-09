#ifndef MOTION_REMOVAL_MOTION_REMOVAL_HPP
#define MOTION_REMOVAL_MOTION_REMOVAL_HPP


#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/video/tracking.hpp>
#include <opencv4/opencv2/optflow.hpp>


namespace motion_removal {

cv::Mat calcOpticalFlowFurnerback(cv::Mat curr, cv::Mat prev);
cv::Mat calcOpticalFlowSparceToDense(cv::Mat curr, cv::Mat prev);

void visualizeOpticalFlow(cv::Mat flow, std::string win_name);

cv::Mat motionRemoval(cv::Mat curr);

}


#endif // MOTION_REMOVAL_MOTION_REMOVAL_HPP
 