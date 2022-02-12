#include "Engine.hpp"
#include <opencv2/opencv.hpp>


int main()
{
    Configurations config;
    config.optBatchSize = {4};

    clock_t start, end;
    double time;


    Engine engine(config);

    bool succ = engine.build(MODEL);
    if(!succ)
    {
        throw runtime_error("Could not built TRT engine");
    }
    succ = engine.loadNetwork();
    if(!succ)
    {
         throw runtime_error("Could not load network");
    }

    const size_t batchSize = 4;
    std::vector<cv::Mat> images;

    const std::string InputImage = "turkish_coffee.jpg";
    //const std::string InputImage = "img.jpg";
    //const std::string InputImage = "images.jpeg";
    auto img = cv::imread(InputImage);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    for (size_t i = 0; i < batchSize; ++i)
    {
        images.push_back(img);
    }

    std::vector<std::vector<float>> featureVectors;
    succ = engine.inference(images, featureVectors);
    if(!succ)
    {
        throw std::runtime_error("Unable to run inference.");
    }

    size_t numIterations = 100;

    for (int i = 0; i < numIterations; ++i)
    {
        featureVectors.clear();
        engine.inference(images, featureVectors);
    }

    std::cout << "Success" << endl;
    
    

    return 0;
}