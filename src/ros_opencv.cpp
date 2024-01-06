#include "../include/motion_removal/ros_opencv.hpp"


RosOpencv::RosOpencv() : Node("motion_removal") {
    rgb_sub_.subscribe(this, "/camera/image_raw");
    depth_sub_.subscribe(this, "/camera/depth/image_raw");

    sync_.reset(new message_filters::Synchronizer<approximate_policy>(approximate_policy(10), rgb_sub_, depth_sub_));
    sync_->registerCallback(std::bind(&RosOpencv::callback, this, std::placeholders::_1, std::placeholders::_2));

    rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw_proc", 10);
    depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/depth/image_raw_proc", 10);
}


RosOpencv::~RosOpencv() {
    cv::destroyAllWindows();
}


cv_bridge::CvImagePtr RosOpencv::rosOpencvRgbConverter(const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb) {
    return cv_bridge::toCvCopy(ros_rgb);
}


cv_bridge::CvImagePtr RosOpencv::rosOpencvDepthConverter(const sensor_msgs::msg::Image::ConstSharedPtr ros_depth) {
    cv_bridge::CvImagePtr depth = cv_bridge::toCvCopy(ros_depth);

    cv::normalize(depth->image, depth->image, 1, 0, cv::NORM_MINMAX);

    return depth;
}


void RosOpencv::callback(const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb, const sensor_msgs::msg::Image::ConstSharedPtr ros_depth) {
    cv_bridge::CvImagePtr rgb =  rosOpencvRgbConverter(ros_rgb);
    cv_bridge::CvImagePtr depth = rosOpencvDepthConverter(ros_depth);

    rgb->image = motion_removal::motionRemoval(rgb->image);

    rgb_publisher_->publish(*rgb->toImageMsg());
    depth_publisher_->publish(*depth->toImageMsg());
#if 0
    RCLCPP_INFO(this->get_logger(), "Messages synced: Callback activated");
#endif
}
