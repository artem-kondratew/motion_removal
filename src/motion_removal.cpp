#include "../include/motion_removal/motion_removal.hpp"


namespace motion_removal {

cv::Mat prev_rgb;
cv::Mat prev_gray;


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


cv::Mat motionRemoval(cv::Mat curr_tgb) {

    cv::Mat curr_gray;
    cv::cvtColor(curr_tgb, curr_gray, cv::COLOR_BGR2GRAY);

    if (prev_rgb.cols == 0) {
        prev_rgb = curr_tgb;
        prev_gray = curr_gray;
        return curr_tgb;
    }

    cv::imshow("curr", curr_tgb);
    cv::imshow("prev", prev_rgb);
    cv::waitKey(20);

    cv::Mat proc = calcOpticalFlowSparceToDense(curr_gray, prev_gray);
    visualizeOpticalFlow(proc, "sparce2dense");

    prev_rgb = curr_tgb;
    prev_gray = curr_gray;

    return curr_tgb;
}

}
