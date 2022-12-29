# DeepStreamTutorials

DeepStreamTutorials contains a series of python apps and notebooks that explore how to run inference using the NVIDIA DeepStream SDK.

I use this repository to learn how to create gstreamer pipeline that include TensorRT models.

DeepStream is working in the GStreamer framework. Here are a few advantages

* Optimize inference speed on NVIDIA GPU using TensorRT.
* Reduce the memory copies in the pipeline.
* Avoid sending data back and forth between CPU and GPU memory.
* Application development can be done in Python or c++.

My main aim is to use [unetTracker models](https://github.com/kevin-allen/unetTracker) within DeepStream to track objects in real-time settings.
I used the apps provided with [DeepStream Python Bindings](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).
I used the example to put together a pipeline that will work for what I want to do.
I got started with my facetrack project.


## To do next

* Use openCV to get largest blob in tensors.
* Save detected blob position in a tracking file.
* Use data from 2 cameras.
* Save tile video when using more than one camera.

## Start the docker container

```
xhost + local:docker

docker run --gpus all -it --net=host --ipc=host -v /tmp/.X11-unix/:/tmp/.X11-unix -v /var/run/docker.sock:/var/run/docker.sock --device /dev/video0   -e DISPLAY=$DISPLAY --privileged -w /opt/nvidia/deepstream/deepstream-6.1 -v /home/kevin/repo:/home/kevin/repo nvcr.io/nvidia/deepstream:6.1.1-devel-mod
```

## Commit your change to a docker container

Find the id of the image you are running and want to save.
```
docker ps
```
Comming the active image to file.
```
docker commit  e0b11bac9560 nvcr.io/nvidia/deepstream:6.1.1-devel-mod
```


## Start a jupyter lab server

```
 cd /home/kevin/repo
 jupyter lab --allow-root --no-browser
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
