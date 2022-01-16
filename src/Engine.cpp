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

using namespace std;
Engine::Engine(const Configurations &config) : m_config(config) {}

Engine::~Engine()
{
}

//Serialize the engine name
string serializeEngineName(const Configurations& config)
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
    engineName = serializeEngineName(m_config);
    if (fileExists(engineName))
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

    //Need to cast enum
    auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network)
    {
        cout << "Network creation failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }

    //Creating the parser
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if(!parser)
    {
        cout << "Parser creation failed!" << endl;
        cout << "Exiting..." << endl;
        return false;
    }
    ifstream file(MODEL, ios::binary | ios::ate);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);
    
    vector<char> buffer(fileSize);
    if(!file.read(buffer.data(), fileSize))
    {
        throw runtime_error("Was not able to parse the model");
    }
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

    //Getting 
    
}


