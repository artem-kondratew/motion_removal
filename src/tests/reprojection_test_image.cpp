#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/calib3d.hpp>

#include <opencv4/opencv2/features2d.hpp>

#include "../../include/motion_removal/motion_removal.hpp"


using namespace motion_removal;


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: ros2 run motion_removal reprojection_test_image <image1> <image2>" << std::endl;
        return 1;
    }

    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);

    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    warp(gray1, gray2);

    cv::waitKey(0);
}
