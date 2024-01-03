#include "../include/motion_removal/ros_opencv.hpp"


RosOpencv::RosOpencv() : Node("motion_removal") {
    rgb_sub_.subscribe(this, "/camera/image_raw");
    depth_sub_.subscribe(this, "/camera/depth/image_raw");

    sync_.reset(new message_filters::Synchronizer<approximate_policy>(approximate_policy(10), rgb_sub_, depth_sub_));
    sync_->registerCallback(std::bind(&RosOpencv::callback, this, std::placeholders::_1, std::placeholders::_2));

    rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw_proc", 10);
    depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/depth/image_raw_proc", 10);
}


cv::Mat proc(cv::Mat img) {
    cv::circle(img, {320, 240}, 50, {255, 0, 255}, -1);
    return img;
}


cv::Mat RosOpencv::rosOpencvRgbConverter(const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb) {
    size_t w = ros_rgb->width;
    size_t h = ros_rgb->height;
    size_t size = w * h;
    size_t channels = 3;

    cv::Mat rgb(h, w, CV_8UC3);

    std::memcpy(rgb.data, ros_rgb->data.data(), sizeof(uint8_t) * size * channels);

    return proc(rgb);
}


float uint8arr_to_float(const uint8_t* data) {
    union {
      float float_variable;
      uint8_t uint8_array[4];
    } un;

    memcpy(un.uint8_array, data, 4);
    return un.float_variable;
}


cv::Mat RosOpencv::rosOpencvDepthConverter(const sensor_msgs::msg::Image::ConstSharedPtr ros_depth) {
    size_t w = ros_depth->width;
    size_t h = ros_depth->height;
    size_t size = w * h;
    size_t channels = 1;

    cv::Mat depth(h, w, CV_32FC1);

    // std::memcpy(depth.data, reinterpret_cast<const float&>(ros_depth.data.data()), sizeof(float) * size * channels);

    // for (size_t i = 0; i < size; i++) {
    //     float pix = uint8arr_to_float(ros_depth.data.data() + i * 4);
    //     depth.data[i] = pix;
    // }

    // cv::imshow("depth", depth);
    // cv::waitKey(20);

    // std::cout << ros_depth.encoding << " " << ros_depth.width << " " << ros_depth.height << " " << ros_depth.data.size() << std::endl;
    // std::cout << (float)depth.data[220000] << " " << (float)ros_depth.data[220000] << std::endl;

    return depth;
}


sensor_msgs::msg::Image::ConstSharedPtr RosOpencv::opencvRosRgbConverter(cv::Mat rgb, const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb) {
    size_t channels = 3;
    size_t size = rgb.cols * rgb.rows;

    auto msg = sensor_msgs::msg::Image();

    msg.header.frame_id = "camera_link_optical";
    msg.header.stamp = this->get_clock()->now();

    msg.height = ros_rgb->height;
    msg.width = ros_rgb->width;

    msg.encoding = ros_rgb->encoding;
    msg.is_bigendian = ros_rgb->is_bigendian;
    msg.step = ros_rgb->step;

    msg.data.resize(size * channels);
    std::memcpy(msg.data.data(), rgb.data, sizeof(uint8_t) * size * channels);

    return std::make_shared<sensor_msgs::msg::Image>(msg);
}


sensor_msgs::msg::Image::ConstSharedPtr RosOpencv::opencvRosDepthConverter(cv::Mat depth, const sensor_msgs::msg::Image::ConstSharedPtr ros_depth) {
    auto msg = sensor_msgs::msg::Image();

    msg.header.frame_id = "camera_link_optical";
    msg.header.stamp = this->get_clock()->now();

    // msg.height = ros_rgb->height;
    // msg.width = ros_rgb->width;

    msg.encoding = ros_depth->encoding;
    msg.is_bigendian = ros_depth->is_bigendian;
    msg.step = ros_depth->step;

    // msg.data.resize(size * channels);
    // std::memcpy(msg.data.data(), rgb.data, sizeof(uint8_t) * size * channels);

    return std::make_shared<sensor_msgs::msg::Image>(msg);
}


void RosOpencv::callback(const sensor_msgs::msg::Image::ConstSharedPtr ros_rgb, const sensor_msgs::msg::Image::ConstSharedPtr ros_depth) {
    cv::Mat rgb =  rosOpencvRgbConverter(ros_rgb);
    cv::Mat depth = rosOpencvDepthConverter(ros_depth);

    auto new_ros_rgb = opencvRosRgbConverter(rgb, ros_rgb);
    auto new_ros_depth = opencvRosDepthConverter(depth, ros_depth);

    rgb_publisher_->publish(*new_ros_rgb.get());
    depth_publisher_->publish(*new_ros_depth.get());

    RCLCPP_INFO(this->get_logger(), "Messages synced: Callback activated");
}
