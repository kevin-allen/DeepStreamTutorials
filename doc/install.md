# Installation

## Docker installation
```
sudo apt-get update
sudo apt-get -y upgrade
sudo ap-get install -y curl
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker <your-user>
sudo reboot
```


## Docker containers download

The recommended way to get going quickly is to use a DeepStream Docker container. 

The `devel` container is for development. It has Graph Composer and build toolchains.

The container for the desktop machine and the Jetson is different.

See [Deepstream docker container page](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html) for more information.

On my desktop machine, I ran

```
docker pull nvcr.io/nvidia/deepstream:6.1.1-devel
```


## Running the container

It allows the container to use the host's X display and GPUs. 

`--gpus all` to use the GPUS

Run `xhost +` before starting the container

```
xhost + local:docker
```

Mount the host's X11 display and map the DISPLAY variable to the docker container: `-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`

For remote execution, install sshfs and openssh-client in the container.

```
docker run --gpus all  --device /dev/video0  -it --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix -v /var/run/docker.sock:/var/run/docker.sock -e DISPLAY=$DISPLAY --privileged -w /opt/nvidia/deepstream/deepstream-6.1  nvcr.io/nvidia/deepstream:6.1.1-devel
```

### Test X display

You can use xclock to test the X display from inside and outside the container.

```
apt-get install -y x11-apps
xclock
```

## Test gstreamer

```
gst-launch-1.0 videotestsrc ! videoconvert ! autovideosink
```

## Test DeepStream

We can use `deepstream-app` to run example of applications using DeepStream. If all goes well, a video should pop up showing the result of inference.


To run inference from a video source

```
deepstream-app -c  /opt/nvidia/deepstream/deepstream-6.1/samples/configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

If you have a usb camera.

```
deepstream-app -c /opt/nvidia/deepstream/deepstream-6.1/samples/configs/deepstream-app/source1_usb_dec_infer_resnet_int8.txt
```

## Install python bindings for DeepStrem

The instructions can be found here: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings

To test the installation

```
cd /opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps/apps/deepstream-test1
python3 deepstream_test_1.py /opt/nvidia/deepstream/deepstream-6.1/samples/streams/sample_720p.h264
```

## Save modifications to a container

This means that we don't have to always redo the installation steps when we launch the container.

```
docker ps -a
docker commit CONTAINER_ID nvcr.io/nvidia/deepstream:6.1.1-devel-mod
docker images
```

