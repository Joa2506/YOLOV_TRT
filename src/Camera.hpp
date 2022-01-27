#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
using namespace cv;

class Camera
{
    private:
        VideoCapture video;
        Mat frame; 
        double fps = 0;
        double fpsAvg = 0;
        double setElapsed;
        double curFPS; // The current fps
        double fpsCalc; // Calculated fps
        double fpsEstimation;

        time_t start, current;
        int numFramesCaptured = 1;

        
    public:
        Camera(VideoCapture video, Mat frame);

        bool run_camera();
};