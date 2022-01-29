#include "Engine.hpp"
#include "Camera.hpp"

class Test
{
    private:
        cv::Mat imgGray, imgBlur;
        cv::Mat frame;
        cv::VideoCapture cap;
    public:
        bool camera_test();
        bool build_engine();
        Test();


};

