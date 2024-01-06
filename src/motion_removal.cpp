#include "../include/motion_removal/motion_removal.hpp"


namespace motion_removal {

cv::Mat prev;


cv::Mat calcDiff(cv::Mat curr) {
    if (prev.rows == 0 || prev.cols == 0) {
        return curr;
    }

    cv::Mat diff = curr - prev;

    return diff;
}


cv::Mat motionRemoval(cv::Mat curr) {

    cv::Mat proc = calcDiff(curr);

    prev = curr;

    cv::imshow("diff", proc);
    cv::waitKey(20);

    return proc;
}

}
