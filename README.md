# YOLOV_TRT

YOLOV_TRT is a project to utilize the yolo model for real time object detection and running inference with the TensorRT framework.
The project will be made to utilize TensorRT framework on the Jetson Xavier architecture

## Installation

Using cmake to install the libraries and dependencies.

```bash
cmake -S src/

make

./engine
```
## TODO

- [x] Add webcam functionality
- [x] Create and test builder function
- [x] Load network function
- [ ] Inference function
- [ ] Test inference on webcam frames
- [ ] Make good tests