# src/CMakeLists.txt


# set the project name
project("Yolov engine")
cmake_minimum_required(VERSION 3.10)

#links libraries
#target_link_libraries(${TARGET_NAME} nvonnxparser)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(TensorRT_DIR "/usr/src/tensorrt")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)
find_package( OpenCV REQUIRED )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_86")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch_86,code=sm_86
	)
endif()


# specify the executable and corresponding source file
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})  #This seems to have added the cuda_runtime api.
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${TensorRT_DIR}/samples/common)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(engine main.cpp Engine.cpp Camera.cpp Test.cpp)

#links libraries
target_link_libraries(engine nvonnxparser)
target_link_libraries(engine nvinfer${NVINFER_LIBRARY})
target_link_libraries(engine ${TensorRT_DIR}/samples/common/logger.cpp)
target_link_libraries(engine ${CUDA_LIBRARIES})
target_link_libraries(engine ${OpenCV_LIBS})