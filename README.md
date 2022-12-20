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

## Goals

* Save video in a file of the captured video
* Use data from 2 cameras.
* Save tile video when using more than one camera.
* Access output tensor from a python app and use openCV to get largest blob.
* Save unet-tracker models as TensorRT engine to speed up start up.

## List of apps

* deepstream-facetrack: Run faceTrack unetTracker model on data from a webcam.

## Documentation

* [Installation](doc/install.md)
* [Example applications](doc/example_apps.md)
* [DeepStream learning material](doc/learning.md)
