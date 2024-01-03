#include "../include/motion_removal/ros_opencv.hpp"


RosOpencv::RosOpencv() : Node("ros_opencv") {
    using namespace std::chrono_literals;
    using std::placeholders::_1;
    rgb_proc_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/rgb_proc", 10);
    rgb_subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/image_raw", 10, std::bind(&RosOpencv::rgbSubCallback, this, _1));
    depth_subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/camera/depth/image_raw", 10, std::bind(&RosOpencv::depthSubCallback, this, _1));
    prev_rgb = cv::Mat();
}


void proc(cv::Mat img) {
    cv::circle(img, {320, 240}, 50, {255, 0, 255}, -1);
}


void RosOpencv::rgbSubCallback(const sensor_msgs::msg::Image& msg) {
    size_t w = msg.width;
    size_t h = msg.height;
    size_t size = w * h;
    size_t channels = 3;

    cv::Mat rgb(h, w, CV_8UC3);

    std::memcpy(rgb.data, msg.data.data(), sizeof(uint8_t) * size * channels);

    proc(rgb);

    prev_rgb = rgb;

    auto pub_msg = sensor_msgs::msg::Image();

    pub_msg.header.frame_id = "camera_link_optical";
    pub_msg.header.stamp = this->get_clock()->now();

    pub_msg.height = msg.height;
    pub_msg.width = msg.width;

    pub_msg.encoding = msg.encoding;
    pub_msg.is_bigendian = msg.is_bigendian;
    pub_msg.step = msg.step;

    pub_msg.data.resize(size * channels);
    std::memcpy(pub_msg.data.data(), rgb.data, sizeof(uint8_t) * size * channels);

    rgb_proc_publisher_->publish(pub_msg);
}

#if 0
float uint8arr_to_float(const uint8_t* data) {
    union {
      float float_variable;
      uint8_t uint8_array[4];
    } un;

    memcpy(un.uint8_array, data, 4);
    return un.float_variable;
}
#endif

void RosOpencv::depthSubCallback(const sensor_msgs::msg::Image& msg) {
    // size_t w = msg.width;
    // size_t h = msg.height;
    // size_t size = w * h;
    // size_t channels = 1;

    // cv::Mat depth(h, w, CV_32FC1);

    //std::memcpy(depth.data, reinterpret_cast<const float&>(msg.data.data()), sizeof(float) * size * channels);

    // for (size_t i = 0; i < size; i++) {
    //     float pix = uint8arr_to_float(msg.data.data() + i * 4);
    //     depth.data[i] = pix;
    // }

    // cv::imshow("depth", depth);
    // cv::waitKey(20);

    // std::cout << msg.encoding << " " << msg.width << " " << msg.height << " " << msg.data.size() << std::endl;
    // std::cout << (float)depth.data[220000] << " " << (float)msg.data[220000] << std::endl;
}
