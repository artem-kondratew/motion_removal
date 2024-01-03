#ifndef MOTION_REMOVAL_ROS_OPENCV_HPP
#define MOTION_REMOVAL_ROS_OPENCV_HPP


#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/time_synchronizer.h>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


class RosOpencv : public rclcpp::Node {
private:
    cv::Mat prev_rgb;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_proc_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;

    // message_filters::TimeSynchronizer<sensor_msgs::msg::Image, 

public:
    RosOpencv();

private:
    void rgbSubCallback(const sensor_msgs::msg::Image& msg);
    void depthSubCallback(const sensor_msgs::msg::Image& msg);
};


#endif // MOTION_REMOVAL_ROS_OPENCV_HPP
