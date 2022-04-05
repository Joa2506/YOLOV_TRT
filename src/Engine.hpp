
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
#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/imgproc.hpp"

using namespace std;

#define MODEL "/home/joakim/Dokumenter/TensorRT/Engine/model/yolov4.onnx"
#define MODEL3 "/home/joakim/Dokumenter/TensorRT/Engine/model/yolov3-10.onnx"
#define MODELTINY "/home/joakim/Dokumenter/TensorRT/Engine/model/tiny-yolov3-11.onnx"
#define MODELONNXMNIST "/home/joakim/Dokumenter/TensorRT/Engine/model/mnist-1.onnx"
#define RESNET "/usr/src/tensorrt/data/resnet50/ResNet50.onnx"
#define MODEL2 "/home/joakim/Dokumenter/TensorRT/Engine/model/yolov2-coco-9.onnx"
#define MODELSSD "/home/joakim/Dokumenter/TensorRT/Engine/model/ssd-12.onnx"

#define MODELJETSON "/home/joakimfj/Documents/TensorRt/YOLOV_TRT/model/yolov4.onnx"
struct Configurations {
    //Using 16 point floats for inference
    bool FP16 = false;
    //Using int8
    bool INT8 = false;
    //Batch size for optimization
    vector<int32_t> optBatchSize;
    // Maximum allowed batch size
    int32_t maxBatchSize = 32;
    //Max GPU memory allowed for the model.
    long int maxWorkspaceSize = 400000000;//
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
        
        //Processes the input image
        bool processInput(Dims& dims, cv::Mat img, float* gpu_input);
        //Verifies the expected output
        bool verifyOutput();

        inline void* safeCudaMalloc(size_t memSize);

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

        samplesCommon::ManagedBuffer m_buffer;


        
        size_t getSizeByDimensions(const nvinfer1::Dims& dims);

    public:

        bool build(string ONNXFILENAME);
        bool loadNetwork();
        bool inference(const vector<cv::Mat> &images, vector<vector<float>>& featureVectors);

        //Constructor and destructor
        Engine(const Configurations& config);
        ~Engine();
};


