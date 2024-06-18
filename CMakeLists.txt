cmake_minimum_required(VERSION 3.8)
project(motion_removal)

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(motion_removal
    src/main.cpp
    include/motion_removal/ros_opencv.hpp
    src/ros_opencv.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(optical_flow_test
    src/tests/optical_flow_test.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(homography_test
    src/tests/homography_test.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(homography_test_image
    src/tests/homography_test_image.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(reprojection_test_image
    src/tests/reprojection_test_image.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(reprojection_test
    src/tests/reprojection_test.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(tum_recorder
    src/tests/tum_recorder.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

add_executable(sparce_test
    src/tests/sparce_test.cpp
    include/motion_removal/motion_removal.hpp
    src/motion_removal.cpp
)

target_link_libraries(motion_removal ${OpenCV_LIBS})
target_link_libraries(optical_flow_test ${OpenCV_LIBS})
target_link_libraries(homography_test ${OpenCV_LIBS})
target_link_libraries(homography_test_image ${OpenCV_LIBS})
target_link_libraries(reprojection_test_image ${OpenCV_LIBS})
target_link_libraries(reprojection_test ${OpenCV_LIBS})
target_link_libraries(tum_recorder ${OpenCV_LIBS})
target_link_libraries(sparce_test ${OpenCV_LIBS})

ament_target_dependencies(motion_removal
    rclcpp
    sensor_msgs
    message_filters
    cv_bridge
)

install(DIRECTORY
    src include config
    DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
    motion_removal
    optical_flow_test
    homography_test
    homography_test_image
    reprojection_test_image
    reprojection_test
    tum_recorder
    sparce_test
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()

