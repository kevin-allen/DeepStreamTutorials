# Example apps

## Getting started with hacking

After running the example apps that were part of the DeepStream framework, 
I moved a few of their application to this repository and started the hacking process.

I had to edit the `config.txt` file that is used to configure the stream. I mainly change relative paths to absolute paths. I also copied the 

```
cp -a /opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps/apps/common /home/kevin/repo/DeepStreamTutorials/apps/
```

After this I tested that an application using a usb camera could work from DeepStreamTutorials.

```
cd DeepStreamTutorials/apps/deepstream_test_1_usb
python3 deepstream_test_1_usb.py /dev/video0
```
