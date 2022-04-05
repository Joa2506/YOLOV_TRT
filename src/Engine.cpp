//CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//TRT
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

//C++
#include <iostream>
#include <fstream>
#include <time.h>
//Own
#include "Engine.hpp"

//OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>



using namespace std;
Engine::Engine(const Configurations &config) : m_config(config) {}

Engine::~Engine()
{
    if(m_stream)
    {
        cudaStreamDestroy(m_stream);
    }
}

//Serialize the engine name
string Engine::serializeEngineName(const Configurations& config)
{
    string name = "trt.engine";

    if(config.FP16)
    {
        name += ".fp16";
    }
    else
    {
        name += ".fp32";
    }
    name += "." + to_string(config.maxBatchSize); + ".";
    for (int i = 0; i < config.optBatchSize.size(); ++i)
    {
        name += to_string(config.optBatchSize[i]);
        if(i != config.optBatchSize.size() - 1)
        {
            name += "_";
        } 
    }

    name += "." + to_string(config.maxWorkspaceSize);
     
    return name;
}


//The builder
bool Engine::build(string ONNXFILENAME)
{
    m_engineName = serializeEngineName(m_config);
    if (fileExists(m_engineName))
    {
        cout << "Engine already exists..." << endl;
        return true;
    }
    //No engine found
    cout << "Building the engine..." << endl;
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if(!builder)
    {
        cout << "Builder creation failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }
    //Set maximum batch size
    builder->setMaxBatchSize(m_config.maxBatchSize);

    cout << "Buider successful!" << endl;
    cout << "Building the Network..." << endl;
    //Need to cast enum
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network)
    {
        cout << "Network creation failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }
    cout << "Network built successfully!" << endl;
    //Creating the parser
    cout << "Building the parser..." << endl;
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if(!parser)
    {
        cout << "Parser creation failed!" << endl;
        cout << "Exiting..." << endl;

        return false;
    }
    cout << "Parser built successfully!" << endl;
    ifstream file(ONNXFILENAME, ios::binary | ios::ate);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    
    vector<char> buffer(fileSize);
    if(!file.read(buffer.data(), fileSize))
    {
        throw runtime_error("Was not able to parse the model");
    }
    cout << "Parsing the parser..." << endl;
    auto parsed = parser->parse(buffer.data(), buffer.size());
    for (size_t i = 0; i < parser->getNbErrors(); i++)
    {
        cout << parser->getError(i)->desc() << endl;
    }
    if(!parsed)
    {
        cout << "Parsing failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }
    cout << "Parsing was successful!" << endl;

    //Getting 
    const auto input = network->getInput(0);
    const auto output = network->getOutput(0);

    m_inputName = input->getName();
    m_outputName = output->getName();
    printf("\n\n%s : %s\n\n", m_inputName, m_outputName);
    m_inputDims = input->getDimensions();
    m_outputDims = output->getDimensions();
    int32_t inputChannel = m_inputDims.d[1];
    int32_t inputHeight = m_inputDims.d[2];
    int32_t inputWidth = m_inputDims.d[3];
    int32_t numberOfInputs = network->getNbInputs();
    // Dims d = network->getInput(0)->getDimensions();
    // printf("\nnumber of Inputs: %d\n", numberOfInputs);
    // printf("\nnumber of dims: %d\n", d.nbDims);

    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    cout << "Configuring the builder..." << endl;
    if(!config)
    {
        cout << "Was not able to build the config" << endl;
        return false;
    }
    cout << "Adding optimization profile..." << endl;
    IOptimizationProfile *defaultProfile = builder->createOptimizationProfile();
    defaultProfile->setDimensions(m_inputName, OptProfileSelector::kMIN, Dims4(1, inputChannel, inputHeight, inputWidth));
    defaultProfile->setDimensions(m_inputName, OptProfileSelector::kOPT, Dims4(1, inputChannel, inputHeight, inputWidth));
    defaultProfile->setDimensions(m_inputName, OptProfileSelector::kMAX, Dims4(m_config.maxBatchSize, inputChannel, inputHeight, inputWidth));
    config->addOptimizationProfile(defaultProfile);

    for (const auto& optBatchSize: m_config.optBatchSize) {
        if (optBatchSize == 1) {
            continue;
        }

        if (optBatchSize > m_config.maxBatchSize) {
            throw std::runtime_error("optBatchSize cannot be greater than maxBatchSize!");
        }

        IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(m_inputName, OptProfileSelector::kMIN, Dims4(1, inputChannel, inputHeight, inputWidth));
        profile->setDimensions(m_inputName, OptProfileSelector::kOPT, Dims4(optBatchSize, inputChannel, inputHeight, inputWidth));
        profile->setDimensions(m_inputName, OptProfileSelector::kMAX, Dims4(m_config.maxBatchSize, inputChannel, inputHeight, inputWidth));
        config->addOptimizationProfile(profile);
    }

    cout << "Optimization profile added" << endl;
    cout << "Setting max workspace size..." << endl;
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, m_config.maxWorkspaceSize);
    cout << "Builder configured successfully" << endl;
    cout << "Making cuda stream..." << endl;
    auto cudaStream = samplesCommon::makeCudaStream();
    if(!cudaStream)
    {
        cout << "Could not create cudaStream." << endl;
        return false;
    }
    cout << "Cuda stream made succsessully" << endl;
    //Setting the profile stream
    config->setProfileStream(*cudaStream);
    cout << "Making serialized model..." << endl;
    unique_ptr<IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
    if(!serializedModel)
    {
        cout << "Could not build serialized model" << endl;
        return false;
    }
    cout << "Model serialized" << endl;

    /*TODO ADD DLA for Tegra 
    */
   
    cout << "Writing serialized model to disk..." << endl;
    //write the engine to disk
    ofstream outfile(m_engineName, ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    cout << "The engine has been built and saved to disk successfully" << endl;

    return true;

    
}

bool Engine::loadNetwork()
{
    ifstream file(m_engineName, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    vector<char> buffer(size);
    cout << "Trying to read engine file..." << endl;
    if(!file.read(buffer.data(), size))
    {
        cout << "Could not read the network from disk" << endl;
        return false;
    }
    cout << "Engine file was read successfully" << endl;
    //Creates a runtime object for running inference
    cout << "Creating a runtime object..." << endl;
    unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if(!runtime)
    {
        cout << "Could not create runtime object" << endl;
        return false;
    }
    cout << "Network object was created successfully" << endl;
    auto ret = cudaSetDevice(m_config.deviceIndex);
    if(ret != 0)
    {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_config.deviceIndex) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    //Let's create the engine
    cout << "Creating the cuda engine..." << endl;
    m_engine = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if(!m_engine)
    {
        cout << "Creating the cuda engine failed" << endl;
        return false;
    }
    

    m_inputName = m_engine->getBindingName(0);
    m_outputName = m_engine->getBindingName(1);
    
    m_inputDims = m_engine->getBindingDimensions(0);

    m_outputDims = m_engine->getBindingDimensions(1);
    cout << "Cuda engine was created successfully" << endl;
    printf("m_inputname == %s\n", m_inputName);
    printf("m_outputname == %s\n", m_outputName);
    
    cout << "Creating execution context..." << endl;
    m_context = shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if(!m_context)
    {
        cout << "Creating the execution context failed" << endl;
        return false;
    }
    cout << "Execution context was created successfully" << endl;
    cout << "Creating CudaStream..." << endl;
    auto cudaRet = cudaStreamCreate(&m_stream);
    printf("%d\n", cudaRet);
    if(cudaRet != 0)
    {
        throw std::runtime_error("Unable to create cuda stream");
    }
    cout << "Cuda stream created successfully!" << endl;
    return true;
}


bool Engine::fileExists(string FILENAME)
{
    ifstream f(FILENAME.c_str());
    return f.good();
}



bool Engine::processInput(Dims& dims, cv::Mat img, float* gpu_input)
{
    cv::Mat frame = img;
    if(frame.empty())
    {
        return false;
    }
    
    
    cv::cuda::GpuMat gpu_frame;
    
    //Upload the frame to the gpu
    printf("Uploading frame to gpu...\n");
    fflush(stdout);
    gpu_frame.upload(frame);
    if(gpu_frame.empty())
    {
        printf("GPU frame is empty\n");
        return false;
    }
    
    printf("Frame uploaded to gpu\n");
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[3];
    auto input_size = cv::Size(input_height, input_width);
    printf("Height = %d\n", input_height);
    printf("Widht = %d\n", input_width);
    printf("Channels = %d\n", channels);
    
    //Resize
    printf("Resizing image\n");
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    if(resized.empty())
    {
        printf("Resized frame is empty\n");
        return false;
    }

    printf("Image resized\n");
    //Normalize
    printf("Normalising image\n");
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f); //FX marks float x CX marks channel x. RGB is x = 3
    if(flt_image.empty())
    {
        printf("FLT is empyt\n");
        return false;
    }
    printf("Converted\n");
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    printf("Subtracted\n");
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    printf("Divided\n");
    //To tensor
    printf("Number of channels %d", channels);
    printf("To Tensor\n");
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < 3; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width*input_height));
    }
    printf("Emplaced back\n");
    fflush(stdout);
    cv::cuda::split(flt_image, chw);
    printf("Made to Tensor\n");
    return true;

}

bool Engine::inference(const std::vector<cv::Mat> &images, std::vector<std::vector<float>>& featureVectors) 
{
    int driverversion;
    cudaRuntimeGetVersion(&driverversion);
    printf("Driver version is: %d\n", driverversion);
    
    cout << "Beginning inference..." << endl;
    std::vector< nvinfer1::Dims > input_dims; // we expect only one input
    std::vector< nvinfer1::Dims > output_dims;
    //vector<void*> buffers(m_engine->getNbBindings());
    vector<void*> buffers(m_engine->getNbBindings());
    printf("Number of bindings: %d\n", m_engine->getNbBindings());
    auto batch_size = images.size();
    for (size_t i = 0; i < m_engine->getNbBindings(); ++i)
    {
        size_t bindingSize = getSizeByDimensions(m_engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], bindingSize);
        if(m_engine->bindingIsInput(i))
        {
            input_dims.emplace_back(m_engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(m_engine->getBindingDimensions(i));
        }
       
    }
    if(input_dims.empty() || output_dims.empty())
    {
            cout << "Expected at least one input and one output for network" << endl;
            return false;
    }
    processInput(input_dims[0], images[0], (float*) buffers[0]);
    m_context->executeV2(buffers.data());
    
    return true;
}


size_t Engine::getSizeByDimensions(const nvinfer1::Dims& dims)
{
    size_t size = 1;

    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    
    return size;
}

inline void* safeCudaMalloc(size_t memSize)
{
    void *deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if(deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

