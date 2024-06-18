#include "../include/motion_removal/ros_opencv.hpp"


int main(int argc, char** argv) {

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotionRemoval>());
    rclcpp::shutdown();
    
    return 0;
}
