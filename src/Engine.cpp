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
    ifstream file(MODEL, ios::binary | ios::ate);
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
    int32_t inputChannel = m_inputDims.d[1];
    int32_t inputHeight = m_inputDims.d[2];
    int32_t inputWidth = m_inputDims.d[3];

    printf("channel : %d\n", inputChannel);
    printf("height : %d\n", inputHeight);
    printf("witdth : %d\n", inputWidth);

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
    config->setMaxWorkspaceSize(m_config.maxWorkspaceSize);
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
    return true;
}



bool Engine::inference(const vector<cv::Mat> &images, vector<vector<float>>& featureVectors)
{
    cout << "Starting inference..." << endl;
    auto dims = m_engine->getBindingDimensions(0);
    auto outputL = m_engine->getBindingDimensions(1).d[1];
    Dims4 inputDims = {static_cast<int32_t>(images.size()), dims.d[1], dims.d[2], dims.d[3]};
    m_context->setBindingDimensions(0, inputDims);

    cout << "Setting dimensions bindings..." << endl;
    if(!m_context->allInputDimensionsSpecified())
    {
        throw runtime_error("Error, not all dimensions specified");
    }

    auto batchSize = static_cast<int32_t>(images.size());
    printf("batchSize: %d\n", batchSize);
    cout << "Preparing batch size.." << endl;
    if(m_previousBatchSize != images.size())
    {
        cout << "Resizing batch size..." << endl;
        m_inputBuffer.hostBuffer.resize(inputDims);
        m_inputBuffer.deviceBuffer.resize(inputDims);

        Dims2 outputDims {batchSize, outputL};

        m_outputBuffer.hostBuffer.resize(outputDims);
        m_outputBuffer.deviceBuffer.resize(outputDims);

        m_previousBatchSize = batchSize;
    }

    auto * hostDataBuffer = static_cast<float*>(m_inputBuffer.hostBuffer.data());
    printf("image size: %ld", images.size());
    printf("\ndims.d[0]: %d | Batch: %d\n", dims.d[0], inputDims.d[0]);
    printf("\ndims.d[1]: %d | inputheight: %d\n", dims.d[1], inputDims.d[1]);
    printf("\ndims.d[2]: %d | inputwidth: %d\n", dims.d[2], inputDims.d[2]);
    printf("\ndims.d[3]: %d | inputChannel: %d\n", dims.d[3], inputDims.d[3]);
    
    // int32_t inputChannel = m_inputDims.d[1];
    // int32_t inputHeight = m_inputDims.d[2];
    // int32_t inputWidth = m_inputDims.d[3];
    cout << "NHCW to NCHW conversion.." << endl;
    for (size_t i = 0; i < images.size(); ++i)
    {
        auto image = images[i];

        image.convertTo(image, CV_32FC3, 1.f / 255.f);   
        cv::subtract(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, cv::noArray(), -1);
        cv::divide(image, cv::Scalar(0.5f, 0.5f, 0.5f), image, 1, -1);    

        //Need to convert to NCHW from NHWC.
        //NHWC: each pixel is stored in RGB order
        //NOTE TO SELF: Found this conversion on stack overflow

        //Test out on Tegra https://stackoverflow.com/questions/36815998/arm-neon-transpose-4x4-uint32 
        int offset = dims.d[1] * dims.d[2] * dims.d[3] * i;
        

        int r = 0, g = 0, b = 0;
        for (size_t j = 0; j < dims.d[1] * dims.d[2] * dims.d[3]; ++j)
        {
            if(j % 3 == 0)
            {
                hostDataBuffer[offset + r++] = *(reinterpret_cast<float*>(image.data) + j);
            }
            else if(j % 3 == 1)
            {
                hostDataBuffer[offset + g++ + 416*416] = *(reinterpret_cast<float*>(image.data) + j);
            }
            else
            {
                hostDataBuffer[offset + b++ + 416*416*2] = *(reinterpret_cast<float*>(image.data) + j);
            }
        // printf("%ld\n", j);
        // fflush(stdout);
        }
        
    }

    cout << "Copying from cpu to gpu..." << endl;
    printf("in host:   %ld\n", m_inputBuffer.hostBuffer.nbBytes());
    printf("in device:   %ld\n", m_inputBuffer.deviceBuffer.nbBytes());
    printf("out host: %ld\n",m_outputBuffer.hostBuffer.nbBytes());
    printf("out device: %ld\n",m_outputBuffer.deviceBuffer.nbBytes());
    //Copying from cpu to gpu
    auto ret = cudaMemcpyAsync(m_inputBuffer.deviceBuffer.data(), m_inputBuffer.hostBuffer.data(), m_inputBuffer.hostBuffer.nbBytes(), cudaMemcpyHostToDevice, m_stream);
    //auto ret = cudaMemcpy(m_inputBuffer.deviceBuffer.data(), m_inputBuffer.hostBuffer.data(), m_inputBuffer.hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
    if(ret != 0)
    {
        cout << "Could not copy from cpu to gpu" << endl;
        fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(ret));
        return false;
    }
    cout << "Creating prediction binding..." << endl;
    vector<void*> predictionBindings = {m_inputBuffer.deviceBuffer.data(), m_outputBuffer.deviceBuffer.data()};

    //Inference
    cout << "Running inference..." << endl;
    bool inference = m_context->enqueueV2(predictionBindings.data(), m_stream, nullptr);
    //bool inference = m_context->executeV2(predictionBindings.data());
    if(!inference)
    {
        cout << "Could not run inference" << endl;
        return inference;
    }
    cout << "Inference was successfull!" << endl;
    //Copy back to cpu
    cout << "Copying from gpu to cpu..." << endl;
    ret = cudaMemcpyAsync(m_outputBuffer.hostBuffer.data(), m_outputBuffer.deviceBuffer.data(), m_outputBuffer.hostBuffer.nbBytes(), cudaMemcpyDeviceToHost, m_stream);
    //ret = cudaMemcpy(m_outputBuffer.hostBuffer.data(), m_outputBuffer.deviceBuffer.data(), m_outputBuffer.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);
    if(ret != 0)
    {
        cout << "Could not copy device from GPU back to cpu" << endl;
        printf("ERROR: %s\n", cudaGetErrorString(ret));
        fflush(stdout);
        return false;
    }
    ret = cudaStreamSynchronize(m_stream);
    if(ret != 0)
    {
        cout << "Unable to synchronize cuda stream!" << endl;
        return false;
    }

    for (size_t i = 0; i < batchSize; ++i)
    {
        vector<float> featureVector;
        featureVector.resize(outputL);

        memcpy(featureVector.data(), reinterpret_cast<const char*>(m_outputBuffer.hostBuffer.data()) +
        i * outputL * sizeof(float), outputL * sizeof(float ));
        featureVectors.emplace_back(std::move(featureVector));

        /* Emplace back
        Appends a new element to the end of the container. The element is constructed through std::allocator_traits::construct,
        which typically uses placement-new to construct the element in-place at the location provided by the container.
        The arguments args... are forwarded to the constructor as std::forward<Args>(args)....
        If the new size() is greater than capacity() then all iterators and references (including the past-the-end iterator) are invalidated. 
        Otherwise only the past-the-end iterator is invalidated.
        */
    }


    return true;
}



bool Engine::fileExists(string FILENAME)
{
    ifstream f(FILENAME.c_str());
    return f.good();
}



bool Engine::processInput()
{
    return false;
}

