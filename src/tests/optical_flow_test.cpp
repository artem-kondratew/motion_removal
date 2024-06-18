#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "../../include/motion_removal/motion_removal.hpp"


void test(std::ifstream* rgb_file, std::string path, std::string method) {
    cv::Mat prev, prev_gray;

    while (true) {
        std::string rgb_name;
        double rgb_t;

        *rgb_file >> rgb_t >> rgb_name;

        if (rgb_name[0] != 'r') {
            return;
        }

        auto start = std::chrono::system_clock::now();

        cv::Mat rgb = cv::imread(path + rgb_name);

        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        if (prev.cols == 0) {
            prev = rgb;
            prev_gray = gray;
            continue;
        }

        if (method == "all") {
            cv::Mat flow_furnerback = motion_removal::calcOpticalFlowFurnerback(prev_gray, gray);
            cv::Mat flow_sparce2dense = motion_removal::calcOpticalFlowSparceToDense(prev_gray, gray);
            motion_removal::visualizeOpticalFlow(flow_furnerback, "furnerback");
            motion_removal::visualizeOpticalFlow(flow_sparce2dense, "sparce2dense");
        }
        if (method == "furnerback") {
            cv::Mat flow_furnerback = motion_removal::calcOpticalFlowFurnerback(prev_gray, gray);
            motion_removal::visualizeOpticalFlow(flow_furnerback, "furnerback");
        }
        if (method == "sparce2dense") {
            cv::Mat flow_sparce2dense = motion_removal::calcOpticalFlowSparceToDense(prev_gray, gray);
            motion_removal::visualizeOpticalFlow(flow_sparce2dense, "sparce2dense");
        }

        prev = rgb;
        prev_gray = gray;

        auto stop = std::chrono::system_clock::now();
        auto delta = (size_t)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << delta << " ms" << std::endl;
    
        cv::imshow("rgb", rgb);
        cv::waitKey(1);
    }
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: ros2 run motion_removal optical_flow_test <path_to tum_folder> <opt flow method> <-l (optional)>" << std::endl;
        std::cout << "opt flow methods:\n    furnerback\n    sparce2dense" << std::endl;
        return 1;
    }

    bool loop = false;

    std::string method = argv[2];
    if (method != "furnerback" && method != "sparce2dense" && method != "all") {
        std::cout << "wrong optical flow method name" << std::endl;
        return 1;
    }

    if (argc == 4) {
        std::string argv2 = argv[3];
        if (argv2 == "-l") {
            loop = true;
        }
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

    size_t i = 0;
    while (true) {
        std::cout << "loop " << i << std::endl;
        test(&rgb_file, path, method);
        rgb_file.close();
        if (!loop) {
            break;
        }
        i++;
        rgb_file.open(path_to_rgb_file);
    }
    
    std::cout << "end of file" << std::endl;
    return 0;
}
