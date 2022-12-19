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
docker run --gpus all -it --net=host --ipc=host -v /tmp/.X11-unix:/tmp/.X11-unix -v /var/run/docker.sock:/var/run/docker.sock -e DISPLAY=$DISPLAY --privileged -w /opt/nvidia/deepstream/deepstream-6.1  nvcr.io/nvidia/deepstream:6.1.1-devel
```

### Test X display

You can use xclock to test the X display from inside and outside the container.

```
apt-get install -y x11-apps
xclock
```


## Save modifications to a container

This means that we don't have to always redo the installation steps when we launch the container.

```
docker ps -a
docker commit CONTAINER_ID nvcr.io/nvidia/deepstream:6.1.1-devel-mod
docker images
```
