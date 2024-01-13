#include "../include/motion_removal/motion_removal.hpp"


namespace motion_removal {

cv::Mat prev;


cv::Mat calcOpticalFlowFurnerback(cv::Mat curr, cv::Mat prev) {
    cv::Mat flow(curr.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    return flow;
}


cv::Mat calcOpticalFlowSparceToDense(cv::Mat curr, cv::Mat prev) {
    cv::Mat flow(curr.size(), CV_32FC2);
    cv::optflow::calcOpticalFlowSparseToDense(prev, curr, flow);
    return flow;
}


void visualizeOpticalFlow(cv::Mat flow, std::string win_name) {
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);

    cv::Mat magnitude, angle, magn_norm;

    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    imshow(win_name, bgr);
}


cv::Mat calcHomography(cv::Mat prev, cv::Mat curr) {
    cv::imshow("prev", prev);
    cv::imshow("curr", curr);

    auto detector = cv::SiftFeatureDetector::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat d1, d2;

    cv::Mat mask = cv::Mat::zeros(prev.size(), CV_8UC1);
    mask.setTo(255);
    
    detector->detectAndCompute(prev, mask, kp1, d1);
    detector->detectAndCompute(curr, mask, kp2, d2);

    auto flann = cv::FlannBasedMatcher();

    std::vector<cv::DMatch> matches;

    flann.match(d1, d2, matches);

    cv::Mat img_with_matches;
    cv::drawMatches(prev, kp1, curr, kp2, matches, img_with_matches);

    cv::imshow("All Matches", img_with_matches);

    std::vector<cv::Point2f> src;
    std::vector<cv::Point2f> dst;

    for (const auto match: matches) {
        src.push_back(kp1[match.queryIdx].pt);
        dst.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(src, dst, cv::LMEDS);

    return H;
}


cv::Mat motionRemoval(cv::Mat curr_rgb) {
    auto start_proc = std::chrono::system_clock::now();

    cv::Mat curr;
    cv::cvtColor(curr_rgb, curr, cv::COLOR_BGR2GRAY);

    if (prev.cols == 0) {
        prev = curr;
        return curr_rgb;
    }

    cv::imshow("curr", curr);
    cv::imshow("prev", prev);
    cv::waitKey(20);

/*-----------------------------------------------------------------------------------------------*/

    auto start_flow = std::chrono::system_clock::now();

    cv::Mat proc = calcOpticalFlowSparceToDense(curr, prev);
    visualizeOpticalFlow(proc, "sparce2dense");

    auto stop_flow = std::chrono::system_clock::now();

/*-----------------------------------------------------------------------------------------------*/

    auto start_homography = std::chrono::system_clock::now();

    cv::Mat H = calcHomography(prev, curr);

    auto stop_homography = std::chrono::system_clock::now();
#if 0
    std::cout << H << std::endl;
#endif

/*-----------------------------------------------------------------------------------------------*/

    prev = curr;

    auto stop_proc = std::chrono::system_clock::now();

    auto flow_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_flow - start_flow).count();
    auto homography_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_homography - start_homography).count();
    auto proc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_proc - start_proc).count();

    std::cout << "opt flow calc duration: " << flow_duration << " ms" << std::endl;
    std::cout << "homography calc duration: " << homography_duration << " ms" << std::endl;
    std::cout << "motion removal duration: " << proc_duration << " ms" << std::endl;
    std::cout << std::endl;

    return curr_rgb;
}

}
