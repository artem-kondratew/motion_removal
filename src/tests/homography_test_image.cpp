#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/calib3d.hpp>

#include <opencv4/opencv2/features2d.hpp>

#include "../../include/motion_removal/motion_removal.hpp"


void dev_homography(cv::Mat gray1, cv::Mat gray2) {

    auto start_detector = std::chrono::system_clock::now();

    cv::imshow("gray1", gray1);
    cv::imshow("gray2", gray2);

    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat d1, d2;

    cv::Mat mask = cv::Mat::zeros(gray1.size(), CV_8UC1);
    mask.setTo(255);
    
    detector->detectAndCompute(gray1, mask, kp1, d1);
    detector->detectAndCompute(gray2, mask, kp2, d2);

    auto flann = cv::FlannBasedMatcher();

    std::vector<cv::DMatch> matches;

    flann.match(d1, d2, matches);

    double max_dist = 0;
    double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (auto i = 0; i < d1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }

    std::cout << "-- Max dist : " << max_dist << std::endl;
    std::cout << "-- Min dist : " << min_dist << std::endl;

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;

    for (auto i = 0; i < d1.rows; i++) {
        if (matches[i].distance <= cv::max(2 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
        }
    }

    auto stop_detector = std::chrono::system_clock::now();

    //-- Draw all matches
    cv::Mat img_matches;
    cv::drawMatches(gray1, kp1, gray2, kp2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //-- Draw only "good" matches
    cv::Mat good_img_matches;
    cv::drawMatches(gray1, kp1, gray2, kp2, good_matches, good_img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("All Matches", img_matches);
    cv::imshow("Good Matches", good_img_matches);

    for (size_t i = 0; i < good_matches.size(); i++) {
        std::cout << "-- Good Match [" << i << "] Keypoint 1: " << good_matches[i].queryIdx << "  -- Keypoint 2: " << good_matches[i].trainIdx << std::endl;
    }

    auto delta_detector = std::chrono::duration_cast<std::chrono::milliseconds>(stop_detector - start_detector).count();
    std::cout << "features detection: " << delta_detector << " ms" << std::endl;

    std::vector<cv::Point2f> src;
    std::vector<cv::Point2f> dst;

    for (const auto match: good_matches) {
        src.push_back(kp1[match.queryIdx].pt);
        dst.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(src, dst, cv::LMEDS);

    std::cout << H << std::endl;

    cv::waitKey(0);
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: ros2 run motion_removal homography_test_image <image1> <image2>" << std::endl;
        return 1;
    }

    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);

    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

#if 0
    dev_homography(gray1, gray2);
#else
    motion_removal::calcHomography(gray1, gray2);
#endif
}