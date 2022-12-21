# DeepStreamTutorials

DeepStreamTutorials contains a series of python apps and notebooks that explore how to run inference using the NVIDIA DeepStream SDK. 

DeepStream is working in the GStreamer framework. Here are a few advantages

* Optimize inference speed.
* No memory copies in the pipeline
* Avoid sending data back and forth between CPU and GPU memory.
* Application development can be done in Python, c++ and in Graph Composer

My main aim is to use unetTracker models within DeepStream to track objects in real-time settings.
I used the apps provided with [DeepStream Python Bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).
I used the example to put together a pipeline that will work for what I want to do.
I got started with my facetrack project.


## To do next

* Use openCV to get largest blob in tensors.
* Save video in a file of the captured video.
* Use data from 2 cameras.
* Save tile video when using more than one camera.


## Start the docker container

```
xhost + local:docker

docker run --gpus all -it --net=host --ipc=host -v /tmp/.X11-unix/:/tmp/.X11-unix -v /var/run/docker.sock:/var/run/docker.sock --device /dev/video0   -e DISPLAY=$DISPLAY --privileged -w /opt/nvidia/deepstream/deepstream-6.1 -v /home/kevin/repo:/home/kevin/repo nvcr.io/nvidia/deepstream:6.1.1-devel-mod
```

## List of apps

#### apps/deepstream-facetrack: 
* Run faceTrack unetTracker model on data from a webcam.
* `python3 deepstream_facetrack.py  dstest_segmentation_config_semantic.txt /dev/video0`

#### utils/onnx_to_trt_engine.py
* Transform an onnx model to a tensorrt engine. The input and output path of the model is hardcoded in the python script.
* This make starting the pipeline much faster.
* Make sure the batch size of the model is set correctly.
* `python3 onnx_to_trt_engine.py`

## Documentation

* [Installation](doc/install.md)
* [Example applications](doc/example_apps.md)
* [DeepStream learning material](doc/learning.md)
