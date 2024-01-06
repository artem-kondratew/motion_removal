cmake_minimum_required(VERSION 3.8)
project(motion_removal)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
    src/main.cpp
    include/motion_removal/ros_opencv.hpp
    src/ros_opencv.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    sensor_msgs
    message_filters
    cv_bridge
)

install(DIRECTORY
    src include
    DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
    ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
