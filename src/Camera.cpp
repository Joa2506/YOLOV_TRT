#include "Camera.hpp"
#include <iostream>
#include <string>

using namespace std;
Camera::Camera(VideoCapture cap, Mat frame)
{
    Camera::video = cap;
    Camera::frame = frame;

}

bool Camera::run_camera()
{
    //Check if camera opens
    if(!video.open(0))
    {
        return false;
    }
    
    time(&start);

    while(true)
    {
        video >> frame;
        if(frame.empty())
        {
            return false;
        }

        //Calculate fps and don't divide by zero
        putText(frame, "FPS: " + to_string(fpsCalc), Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 2.1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        imshow("Img", frame);

        time(&current);
        setElapsed = difftime(current, start);
        if(setElapsed != 0)
        {
            fpsCalc = numFramesCaptured/setElapsed;
            numFramesCaptured++;
            fpsEstimation = fpsEstimation + fpsCalc;
            fpsAvg = fpsEstimation / numFramesCaptured;
        }
        printf("Captured FPS: %f\n", fpsCalc);


        //TODO: Run inference on the frames in here somewhre
        
        if(waitKey(10) == 27)
        {
            return true;
        }

    }   
    
}