#ifndef MOTION_REMOVAL_MOTION_REMOVAL_HPP
#define MOTION_REMOVAL_MOTION_REMOVAL_HPP


#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


namespace motion_removal {

cv::Mat calcDiff(cv::Mat curr);

cv::Mat motionRemoval(cv::Mat curr);

}


#endif // MOTION_REMOVAL_MOTION_REMOVAL_HPP
 