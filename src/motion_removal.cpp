#include "../include/motion_removal/motion_removal.hpp"


namespace motion_removal {

cv::Mat prev;


/// @brief Computes a dense optical flow using the Gunnar Farneback's algorithm.
/// @param prev previous single-channel grayscale frame.
/// @param curr current single-channel grayscale frame.
/// @return flow image with flow vectors (u, v), where u, v are offsets in pixels.
cv::Mat calcOpticalFlowFurnerback(cv::Mat prev, cv::Mat curr) {
    cv::Mat flow(curr.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(prev, curr, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    return flow;
}


/// @brief Computes dense optical flow using PyrLK sparse matches interpolation.
/// @param prev previous single-channel grayscale frame.
/// @param curr current single-channel grayscale frame.
/// @return flow image with flow vectors (u, v), where u, v are offsets in pixels.
cv::Mat calcOpticalFlowSparceToDense(cv::Mat prev, cv::Mat curr) {
    cv::Mat flow(curr.size(), CV_32FC2);
    cv::optflow::calcOpticalFlowSparseToDense(prev, curr, flow);
    return flow;
}


/// @brief Visualizes flow image.
/// @param flow 2-channel flow image.
/// @param win_name OpenCV window name.
/// @return BGR 3-channel flow image visualization.
cv::Mat visualizeOpticalFlow(cv::Mat flow, std::string win_name) {
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

    return bgr;
}


/// @brief Finds good matches from detector for homography matrix estimation.
/// @param matches all matches.
/// @param d1 descriptors from previous image.
/// @return good matches vector.
std::vector<cv::DMatch> findGoodMatches(std::vector<cv::DMatch> matches, cv::Mat d1) {
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

    // std::cout << "-- Max dist : " << max_dist << std::endl;
    // std::cout << "-- Min dist : " << min_dist << std::endl;

    std::vector<cv::DMatch> good_matches;

    for (auto i = 0; i < d1.rows; i++) {
        if (matches[i].distance <= cv::max(2 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
        }
    }

    return good_matches;
}


/// @brief Computes homography matrix from two single-channel grayscale images.
/// @param prev previous single-channel grayscale frame.
/// @param curr current single-channel grayscale frame.
/// @param use_good_matches flag for using only good matches from SIFT detector.
/// @return 3x3 homography matrix.
cv::Mat calcHomography(cv::Mat prev, cv::Mat curr, bool use_good_matches) {
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

    if (use_good_matches && matches.size() > 4) {
        auto good_matches = findGoodMatches(matches, d1);
        if (good_matches.size() >= 4) {
            matches = good_matches;
        }
    }

    std::vector<cv::Point2f> src;
    std::vector<cv::Point2f> dst;

    for (const auto match: matches) {
        src.push_back(kp1[match.queryIdx].pt);
        dst.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(src, dst, cv::LMEDS);

    return H;
}


/// @brief Computes warped image from previous image via homography matrix.
/// @param prev previous single-channel grayscale frame.
/// @param curr current single-channel grayscale frame.
/// @return warped single-channel grayscale image.
cv::Mat sparce(cv::Mat prev, cv::Mat curr) {
    auto detector = cv::SiftFeatureDetector::create();

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat d1, d2;

    cv::Mat mask = cv::Mat::zeros(prev.size(), CV_8UC1);
    mask.setTo(255);
    
    detector->detectAndCompute(prev, mask, kp1, d1);
    detector->detectAndCompute(curr, mask, kp2, d2);

    auto flann = cv::FlannBasedMatcher();

    std::vector<cv::DMatch> matches, h_matches;

    flann.match(d1, d2, matches);

    if (matches.size() > 4) {
        auto good_matches = findGoodMatches(matches, d1);
        h_matches = (good_matches.size() >= 4) ? good_matches : matches;
    }

    std::vector<cv::Point2f> src;
    std::vector<cv::Point2f> dst;

    for (const auto match: h_matches) {
        src.push_back(kp1[match.queryIdx].pt);
        dst.push_back(kp2[match.trainIdx].pt);
    }

    std::vector<cv::Point2f> flow_src;
    std::vector<cv::Point2f> flow_dst;

    for (const auto match: matches) {
        flow_src.push_back(kp1[match.queryIdx].pt);
        flow_dst.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(src, dst, cv::LMEDS);

    cv::Mat curr_warped;
    cv::warpPerspective(curr, curr_warped, H, curr.size());

    cv::Mat status, err;
    cv::calcOpticalFlowPyrLK(prev, curr_warped, flow_src, flow_dst, status, err);

    std::vector<cv::Point2f> good_dst;
    std::vector<cv::Point2f> bad_dst;
    std::vector<float> errors;
    float s = 0;

    for (size_t i = 0; i < flow_dst.size(); i++) {
        float dx = flow_dst[i].x - flow_src[i].x;
        float dy = flow_dst[i].y - flow_src[i].y;

        float e = std::sqrt(dx * dx + dy * dy);
        errors.push_back(e);
        s += e;
    }

    float average_e = s / flow_dst.size();

    for (size_t i = 0; i < flow_dst.size(); i++) {
        if (errors[i] > average_e) {
            bad_dst.push_back(flow_dst[i]);
        }
        else {
            good_dst.push_back(flow_dst[i]);
        }
    }

    cv::Mat img_matches;
    cv::cvtColor(curr, img_matches, cv::COLOR_GRAY2BGR);

    for (auto pt: bad_dst) {
        cv::circle(img_matches, {(int)pt.x, (int)pt.y}, 4, {0, 0, 255});
    }

    for (auto pt: good_dst) {
        cv::circle(img_matches, {(int)pt.x, (int)pt.y}, 4, {0, 255, 0});
    }

    cv::imshow("matches", img_matches);

    return cv::Mat();
}


/// @brief Computes warped image from previous image via homography matrix.
/// @param prev previous single-channel grayscale frame.
/// @param curr current single-channel grayscale frame.
/// @return warped single-channel grayscale image.
cv::Mat warp(cv::Mat prev, cv::Mat curr) {
    cv::Mat opt_flow = calcOpticalFlowSparceToDense(prev, curr);

    cv::Mat loc_prev = cv::Mat::zeros(curr.size(), CV_32FC2);
    cv::Mat loc_curr = cv::Mat::zeros(curr.size(), CV_32FC2);
    for (int y = 0; y < curr.rows; y++) {
        for (int x = 0; x < curr.cols; x++) {
            cv::Point2f& flow_pt = opt_flow.at<cv::Point2f>(y, x);
            cv::Point2f& prev_pt = loc_prev.at<cv::Point2f>(y, x);
            cv::Point2f& curr_pt = loc_curr.at<cv::Point2f>(y, x);

            // flow_pt.x = std::max(1.0f, std::min(static_cast<float>(curr.cols - 2), flow_pt.x));
            // flow_pt.y = std::max(1.0f, std::min(static_cast<float>(curr.rows - 2), flow_pt.y));

            curr_pt.x = x;
            curr_pt.y = y;

            prev_pt.x = curr_pt.x - flow_pt.x;
            prev_pt.y = curr_pt.y - flow_pt.y;
        }
    }

    cv::Mat H = calcHomography(curr, prev, true);
    // cv::Mat H = calcHomography(prev, curr, true);
    // std::cout << H << std::endl;

    cv::Mat loc_curr_warped;
    cv::Mat loc_prev_warped;
    cv::warpPerspective(loc_curr, loc_curr_warped, H, curr.size());
    // cv::warpPerspective(loc_prev, loc_prev_warped, H, curr.size());

    cv::Mat err = loc_prev - loc_curr_warped;
    // cv::Mat err = loc_prev_warped - loc_curr;
    cv::Mat second_norm = cv::Mat::zeros(curr.size(), CV_32FC1);

    for (int y = 0; y < curr.rows; y++) {
        for (int x = 0; x < curr.cols; x++) {
            cv::Point2f& err_pt = err.at<cv::Point2f>(y, x);
            second_norm.at<float>(y, x) = std::sqrt(err_pt.x * err_pt.x + err_pt.y * err_pt.y);
        }
    }

    double min, max;

    size_t roi_x = 50;
    size_t roi_y = 50;
    size_t roi_w = curr.cols - roi_x * 2;
    size_t roi_h = curr.rows - roi_y * 2;

    cv::Mat roi = second_norm(cv::Rect(roi_x, roi_y, roi_w, roi_h)).clone();

    float avg = 0;
    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            avg += roi.at<float>(y, x);
        }
    }

    avg = avg / (roi.rows * roi.cols);

    cv::minMaxLoc(roi, &min, &max);
    std::cout << "sn: ";
    std::cout << "min = " << min << " max = " << max << " avg = " << avg << std::endl;
    
    // cv::Mat phi;
    // cv::exp(roi, phi);

    // cv::Mat inf_mask = (phi == std::numeric_limits<float>::infinity());
    // phi.setTo(std::numeric_limits<float>::max(), inf_mask);

    // cv::Mat phi_normalized;
    // cv::normalize(phi, phi_normalized, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    // cv::threshold(phi_normalized, phi_normalized, 0, 0, cv::THRESH_TOZERO);
    // cv::threshold(phi_normalized, phi_normalized, 1, 1, cv::THRESH_TRUNC);

    // cv::Mat log;
    // cv::log(phi_normalized / (1 - phi_normalized), log);
    
    // cv::minMaxLoc(log, &min, &max);
    // std::cout << "log: ";
    // std::cout << "min = " << min << " max = " << max << std::endl << std::endl;

    // cv::Mat mask = cv::Mat::zeros(log.size(), CV_8UC1);
    // for (int y = 0; y < mask.rows; y++) {
    //     for (int x = 0; x < mask.cols; x++) {
    //         float ln = phi_normalized.at<float>(y, x);
    //         mask.at<uchar>(y, x) = (ln >= 0.01) ? 255 : 0;
    //     }
    // }

    cv::Mat second_norm_normalized;
    cv::normalize(roi, second_norm_normalized, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    cv::threshold(second_norm_normalized, second_norm_normalized, 0, 0, cv::THRESH_TOZERO);
    cv::threshold(second_norm_normalized, second_norm_normalized, 1, 1, cv::THRESH_TRUNC);

    cv::Mat log;
    cv::log(second_norm_normalized / (1 - second_norm_normalized), log);

    cv::minMaxLoc(log, &min, &max);
    std::cout << "log: ";
    std::cout << "min = " << min << " max = " << max << std::endl << std::endl;

    cv::Mat mask = cv::Mat::zeros(log.size(), CV_8UC1);
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            float ln = log.at<float>(y, x);
            mask.at<uchar>(y, x) = (ln >= 0) ? 255 : 0;
        }
    }

    cv::Mat second_norm_visual;
    cv::normalize(second_norm, second_norm_visual, 0.0, 1.0, cv::NORM_MINMAX);

    visualizeOpticalFlow(opt_flow, "flow");
    cv::imshow("second_norm", second_norm_visual);

    cv::imshow("mask", mask);

    return mask;
}


/// @brief Main function. Includes pipeline for motion removal.
/// @param curr_rgb current rgb frame from depth camera.
/// @return rgb frame without moving objects.
cv::Mat motionRemoval(cv::Mat curr_rgb) {
    cv::imshow("rgb", curr_rgb);
    cv::waitKey(25);
    return curr_rgb;
//     auto start_proc = std::chrono::system_clock::now();

//     cv::Mat curr;
//     cv::cvtColor(curr_rgb, curr, cv::COLOR_BGR2GRAY);

//     if (prev.cols == 0) {
//         prev = curr;
//         return curr_rgb;
//     }

//     cv::imshow("curr", curr);
//     cv::imshow("prev", prev);
//     cv::waitKey(20);

// /*-----------------------------------------------------------------------------------------------*/

//     auto start_flow = std::chrono::system_clock::now();

//     cv::Mat proc = calcOpticalFlowSparceToDense(prev, curr);
//     visualizeOpticalFlow(proc, "sparce2dense");

//     auto stop_flow = std::chrono::system_clock::now();

// /*-----------------------------------------------------------------------------------------------*/

//     auto start_homography = std::chrono::system_clock::now();

//     cv::Mat H = calcHomography(prev, curr, false);

//     auto stop_homography = std::chrono::system_clock::now();
// #if 0
//     std::cout << H << std::endl;
// #endif

// /*-----------------------------------------------------------------------------------------------*/

//     prev = curr;

//     auto stop_proc = std::chrono::system_clock::now();

//     auto flow_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_flow - start_flow).count();
//     auto homography_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_homography - start_homography).count();
//     auto proc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_proc - start_proc).count();

//     std::cout << "opt flow calc duration: " << flow_duration << " ms" << std::endl;
//     std::cout << "homography calc duration: " << homography_duration << " ms" << std::endl;
//     std::cout << "motion removal duration: " << proc_duration << " ms" << std::endl;
//     std::cout << std::endl;

//     return curr_rgb;
}

}
