#include <fstream>
#include <string>

#include "../include/motion_removal/motion_removal.hpp"


void test(std::ifstream* rgb_file, std::string path) {
    cv::Mat prev, prev_gray;

    while (true) {

        std::string rgb_name;
        double rgb_t;

        *rgb_file >> rgb_t >> rgb_name;

        if (rgb_name[0] != 'r') {
            return;
        }

        cv::Mat rgb = cv::imread(path + rgb_name);

        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        if (prev.cols == 0) {
            prev = rgb;
            prev_gray = gray;
            continue;
        }

        cv::Mat flow = motion_removal::calcOpticalFlow(gray, prev_gray);
        motion_removal::visualizeOpticalFlow(flow);

        prev = rgb;
        prev_gray = gray;
    
        cv::imshow("rgb", rgb);
        cv::waitKey(1);
    }
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: ros2 run motion_removal optical_flow_test <path_to tum_folder>" << std::endl;
        return 1;
    }

    bool loop = false;

    if (argc == 3) {
        std::string argv2 = argv[2];
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
        test(&rgb_file, path);
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
