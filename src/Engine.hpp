
//TRT headers
#include <NvInfer.h>

//for cp
#include <buffers.h>
#include <memory>

//Own libaries
#include "Logger.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

#define MODEL "/home/joakim/Dokumenter/TensorRT/Engine/model/yolov4.onnx"
#define MODEL2 "/home/joakim/Dokumenter/TensorRT/Engine/model/yolov3-10.onnx"
#define MODELTINY "/home/joakim/Dokumenter/TensorRT/Engine/model/tiny-yolov3-11.onnx"
#define MODELONNXMNIST "/home/joakim/Dokumenter/TensorRT/Engine/model/mnist-1.onnx"
#define RESNET "/usr/src/tensorrt/data/resnet50/ResNet50.onnx"
#define MODELYOLOV2 "/home/joakim/Dokumenter/TensorRT/Engine/model/yolov2-coco-9.onnx"

#define MODELJETSON "/home/joakimfj/Documents/TensorRt/YOLOV_TRT/model/yolov4.onnx"
struct Configurations {
    //Using 16 point floats for inference
    bool FP16 = false;
    //Using int8
    bool INT8 = false;
    //Batch size for optimization
    vector<int32_t> optBatchSize;
    // Maximum allowed batch size
    int32_t maxBatchSize = 16;
    //Max GPU memory allowed for the model.
    long int maxWorkspaceSize = 4294967296;//
    //GPU device index number, might be useful for more Tegras in the future
    int deviceIndex = 0;
    // DLA
    int dlaCore = 0;

};

class Engine
{
    private:
        //Checks if the engine with the similar filename already exists
        bool engineExists(string FILENAME);
        bool fileExists(string FILENAME);
        //Serializes the engine name based on the configurations added by the user
        string serializeEngineName(const Configurations& config);
        
        //Might need to process the data that we runs inference on
        bool processInput();
        //Verifies the expected output
        bool verifyOutput();

        //Engine for running inference
        shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        //Execution context for inference
        shared_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

        //ILogger
        Logger m_logger;

        //Dimensions of inputs and outputs
        Dims m_inputDims;
        Dims m_outputDims;

        const Configurations& m_config;
        //Cuda stream
        cudaStream_t m_stream = nullptr;

        //Engine name
        string m_engineName;

        size_t m_previousBatchSize;
        //
        const char * m_inputName;
        const char * m_outputName;

        samplesCommon::ManagedBuffer m_inputBuffer;
        samplesCommon::ManagedBuffer m_outputBuffer;


    public:

        bool build(string ONNXFILENAME);
        bool loadNetwork();
        bool inference(const vector<cv::Mat> &images, vector<vector<float>>& featureVectors);

        //Constructor and destructor
        Engine(const Configurations& config);
        ~Engine();
};


