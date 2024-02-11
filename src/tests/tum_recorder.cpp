#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


void test(std::ifstream* rgb_file, std::string path) {
    cv::Mat prev, prev_gray;

    cv::Mat frames[743];

    int cnt = 0;

    while (true) {
        std::string rgb_name;
        double rgb_t;

        *rgb_file >> rgb_t >> rgb_name;

        if (rgb_name[0] != 'r') {
            break;
        }

        cv::Mat rgb = cv::imread(path + rgb_name);
        frames[cnt] = rgb;
        cnt++;
        
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        if (prev.cols == 0) {
            prev = rgb;
            prev_gray = gray;
            continue;
        }

        prev = rgb;
        prev_gray = gray;
    
        cv::imshow("rgb", rgb);
        cv::waitKey(1);
        std::cout << "cnt = " << cnt << std::endl;
    }

    std::cout << "start recording" << std::endl;

    auto writer = cv::VideoWriter("tum_dataset.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 40, {640, 480});
    for (int i = 0; i < 743; i++) {
        writer.write(frames[i]);
    }
    writer.release();
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: ros2 run motion_removal tum_recorder <path_to tum_folder>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    path = path[path.size()-1] == '/' ? path : path + '/';

    std::string path_to_rgb_file = path + "rgb.txt";
    std::ifstream rgb_file;
    rgb_file.open(path_to_rgb_file);

    if (!rgb_file.is_open()) {
        std::cout << "error while opening rgb_file with path: " << path_to_rgb_file << std::endl;
        return 1;
    }

    test(&rgb_file, path);
    rgb_file.close();
    
    std::cout << "end of file" << std::endl;
    return 0;
}
