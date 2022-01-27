#include "Camera.hpp"


int main()
{
    cv::Mat imgGray, imgBlur;
    cv::Mat frame;
    cv::VideoCapture cap;
    Camera cam(cap, frame);
    cam.run_camera();
}