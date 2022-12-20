# DeepStreamTutorials

DeepStreamTutorials contains a series of python apps and notebooks that explore how to run inference using the NVIDIA DeepStream SDK. 

DeepStream is working in the GStreamer framework. Here are a few advantages

* Optimize inference speed.
* No memory copies in the pipeline
* Avoid sending data back and forth between CPU and GPU memory.
* Application development can be done in Python, c++ and in Graph Composer

My main aim was to be able to run unetTracker models within DeepStream. I got started with my facetrack project.

## List of apps

* deepstream-facetrack: Run faceTrack unetTracker model on data from a webcam.


## Documentation

* [Installation](doc/install.md)
* [Example applications](doc/example_apps.md)
* [DeepStream learning material](doc/learning.md)
