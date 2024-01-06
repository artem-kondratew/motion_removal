#ifndef MOTION_REMOVAL_ROS_OPENCV_HPP
#define MOTION_REMOVAL_ROS_OPENCV_HPP


#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include "motion_removal.hpp"


class RosOpencv : public rclcpp::Node {
private:
    using approximate_policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;

    std::unique_ptr<message_filters::Synchronizer<approximate_policy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_publisher_;

    cv_bridge::CvImagePtr rosOpencvRgbConverter(const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb);
    cv_bridge::CvImagePtr rosOpencvDepthConverter(const sensor_msgs::msg::Image::ConstSharedPtr ros_depth);

public:
    RosOpencv();
    ~RosOpencv();

    void callback(const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb, const sensor_msgs::msg::Image::ConstSharedPtr ros_depth);
};


#endif // MOTION_REMOVAL_ROS_OPENCV_HPP
