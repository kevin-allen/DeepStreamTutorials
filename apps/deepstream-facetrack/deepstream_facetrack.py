#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys

sys.path.append('../')
import gi
import math
import ctypes

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import cv2
import pyds
import numpy as np
import os.path
from os import path

MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
COLORS = [[128, 128, 64], [0, 0, 128], [0, 128, 128], [128, 0, 0],
          [128, 0, 128], [128, 128, 0], [0, 128, 0], [0, 0, 64],
          [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64],
          [128, 0, 192], [128, 128, 128]]


def map_mask_as_display_bgr(mask):
    """ Assigning multiple colors as image output using the information
        contained in mask. (BGR is opencv standard.)
    """
    # getting a list of available classes
    m_list = list(set(mask.flatten()))

    shp = mask.shape
    bgr = np.zeros((shp[0], shp[1], 3))
    for idx in m_list:
        bgr[mask == idx] = COLORS[idx]
    return bgr


def seg_src_pad_buffer_probe(pad, info, u_data):
    """
    Function to access inference data output

    There is two types of meta data
    - segmentation (2D array with integer representing the segmentation image (classes id))
    - tensor (raw output of the TensorRT inference engine)
    """

    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None: # I presume we are looping over the frames in the batch
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.

            # https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number = frame_meta.frame_num
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break


            
            if user_meta and user_meta.base_meta.meta_type == \
                    pyds.NVDSINFER_SEGMENTATION_META:
                try:
                    # Note that seg_user_meta.user_meta_data needs a cast to
                    # pyds.NvDsInferSegmentationMeta
                    # The casting is done by pyds.NvDsInferSegmentationMeta.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone.

            
                    #https://docs.nvidia.com/metropolis/deepstream/sdk-api/structNvDsInferSegmentationMeta.html
                    segmeta = pyds.NvDsInferSegmentationMeta.cast(user_meta.user_meta_data)
                    
                except StopIteration:
                    break
                # Retrieve mask data in the numpy format from segmeta
                # Note that pyds.get_segmentation_masks() expects object of
                # type NvDsInferSegmentationMeta

                #print(tensor_meta.gpu_id)
                
                masks = pyds.get_segmentation_masks(segmeta)
                masks = np.array(masks, copy=True, order='C')
                #print("frame_number:", frame_number, ", mask shape:", masks.shape, ", mask range:", masks.min(),masks.max())
                # map the obtained masks to colors of 2 classes.
                #frame_image = map_mask_as_display_bgr(masks)
                #cv2.imwrite(folder_name + "/" + str(frame_number) + ".jpg", frame_image)



                
            # https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#tensor-metadata
            if user_meta and user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                try:
                #    print("Frame:", frame_number, ", Found our tensor output")
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)                   
                 #   print("retrieved")
                    

                except StopIteration:
                    break

               # print("tensor_meta.num_output_layers:", tensor_meta.num_output_layers)
                # https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsInfer/NvDsInferLayerInfo.html
                output_layers_info = pyds.get_nvds_LayerInfo(tensor_meta,0)
                

                # https://docs.nvidia.com/metropolis/deepstream/python-api/PYTHON_API/NvDsInfer/NvDsInferDims.html#pyds.NvDsInferDims
                #print("dataType:",output_layers_info.dataType,
                #      ", inferDims.numDims:", output_layers_info.inferDims.numDims,
                #      ", inferDims.numElements:", output_layers_info.inferDims.numElements,
                #      ", inferDims.d:", output_layers_info.inferDims.d,
                #      ", layerName:",output_layers_info.layerName)

                shp = output_layers_info.inferDims.d[:output_layers_info.inferDims.numDims]
                ptr = ctypes.cast(pyds.get_ptr(output_layers_info.buffer), ctypes.POINTER(ctypes.c_float))
                v = np.ctypeslib.as_array(ptr, shape=shp)
                #print("max:", v.reshape(4,-1).max(axis=1))
                

            try:
                l_user = l_user.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK



def main(args):
    # Check input arguments
    if len(args) != 3:
        sys.stderr.write("usage:python3 %s config_file <dev/video0>\n" % args[0])
        sys.exit(1)

    config_file = args[1]
    print(config_file)
    num_sources = 1
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")



        
    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")


    print("Creating Video Converter \n")

    # Adding videoconvert -> nvvideoconvert as not all
    # raw formats are supported by nvvideoconvert;
    # Say YUYV is unsupported - which is the common
    # raw format for many logi usb cams
    # In case we have a camera with raw format supported in
    # nvvideoconvert, GStreamer plugins' capability negotiation
    # shall be intelligent enough to reduce compute by
    # videoconvert doing passthrough (TODO we need to confirm this)


    # videoconvert to make sure a superset of raw formats are supported
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")


        
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Create segmentation for primary inference
    seg = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    if not seg:
        sys.stderr.write("Unable to create primary inferene\n")


    tee = Gst.ElementFactory.make("tee", "tee1")
    if not tee:
        sys.stderr.write(" Unable to create tee\n")


    queue1 = Gst.ElementFactory.make("queue", "q1")
    if not queue1:
        sys.stderr.write(" Unable to create queue1\n")

    queue2 = Gst.ElementFactory.make("queue", "q2")
    if not queue2:
        sys.stderr.write(" Unable to create queue2\n")
    


    # Create nvsegvisual for visualizing segmentation
    #nvsegvisual = Gst.ElementFactory.make("nvsegvisual", "nvsegvisual")
    #if not nvsegvisual:
    #    sys.stderr.write("Unable to create nvsegvisual\n")
    #sink_seg = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer-seg")
    #if not sink_seg:
    #    sys.stderr.write(" Unable to create egl sink_seg \n")

    
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")


    sink_osd = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer-osd")
    if not sink_osd:
        sys.stderr.write(" Unable to create egl sink_seg \n")

    
    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")


    # elements needed to save a file
    # nvvidconv !  'video/x-raw(memory:NVMM),width=480, height=480' ! nvv4l2h264enc  ! h264parse ! qtmux ! filesink  location="filename_h264.mp4"

    
    nvvidconvposttee = Gst.ElementFactory.make("nvvideoconvert", "convertor_posttee")
    if not nvvidconvposttee:
        sys.stderr.write(" Unable to create Nvvideoconvert posttee \n")

    caps_vidconvposttee = Gst.ElementFactory.make("capsfilter", "nvmm_caps_posttee")
    if not caps_vidconvposttee:
        sys.stderr.write(" Unable to create capsfilter posttee \n")

    nvvenc = Gst.ElementFactory.make("nvv4l2h264enc", "enc")
    if not nvvenc:
        sys.stderr.write(" Unable to create nvvenc\n")

    parse = Gst.ElementFactory.make("h264parse", "parse")
    if not nvvenc:
        sys.stderr.write(" Unable to create parse\n")
    
    mux =  Gst.ElementFactory.make("matroskamux", "mux")
    if not mux:
        sys.stderr.write(" Unable to create mux\n")
    
    file_sink =  Gst.ElementFactory.make("filesink", "fs")
    if not file_sink:
        sys.stderr.write(" Unable to create file_sink\n")
    
    
        
    print("Playing video %s " % args[2])


    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', args[2])

    
    streammux.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_DEVICE))
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000)
    streammux.set_property('live-source', 1) # Essential to get more than 1 Hz
    
    seg.set_property('config-file-path', config_file)
    pgie_batch_size = seg.get_property("batch-size")
    if pgie_batch_size != num_sources:
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size,
              " with number of sources ", num_sources,
              " \n")
        seg.set_property("batch-size", num_sources)
    #nvsegvisual.set_property('batch-size', num_sources)
    #nvsegvisual.set_property('width', 640)
    #nvsegvisual.set_property('height', 480)

    
    caps_vidconvposttee.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM),width=640, height=480'))
    
    sink_osd.set_property("qos", 0)


    nvvenc.set_property('bitrate', 4000000)

    
    file_sink.set_property("location", "test.mp4")

    print("Adding elements to Pipeline \n")


    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)

    pipeline.add(streammux)
    pipeline.add(seg)
    pipeline.add(tee)
    pipeline.add(queue1)
    pipeline.add(queue2)
    #pipeline.add(nvsegvisual)
    # for display
    pipeline.add(nvosd)
    pipeline.add(sink_osd)

    # for file 
    pipeline.add(nvvidconvposttee)
    pipeline.add(caps_vidconvposttee)
    pipeline.add(nvvenc)
    pipeline.add(parse)
    pipeline.add(mux)
    pipeline.add(file_sink)

    
    #pipeline.add(sink_seg)
    if is_aarch64():
        pipeline.add(transform)




    # we link the elements together
    # file-source -> jpeg-parser -> nvv4l2-decoder ->
    # nvinfer -> nvsegvisual -> sink
    print("Linking elements in the Pipeline \n")


    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)


    streammux.link(seg)
    seg.link(tee)


    tee_osd_pad = tee.get_request_pad('src_%u')
    tee_file_pad = tee.get_request_pad("src_%u")    
    if not tee_osd_pad or not tee_file_pad:
        sys.stderr.write("Unable to get request tee pads\n")

    sink_queue1_pad = queue1.get_static_pad('sink')
    if not sink_queue1_pad:
        sys.stderr.write("Unable to get request sink_queue1 pad\n")

    sink_queue2_pad = queue2.get_static_pad('sink')
    if not sink_queue2_pad:
        sys.stderr.write("Unable to get request sink_queue2 pad\n")
        

        
    #sink_nvsegvisual_pad = nvsegvisual.get_static_pad('sink')
    #if not sink_nvsegvisual_pad:
    #    sys.stderr.write("Unable to get request nvsegvisual pad\n")

    tee_osd_pad.link(sink_queue1_pad)
    queue1.link(nvosd)
    #tee_seg_pad.link(sink_nvsegvisual_pad)
    
    
    #tee.link(nvsegvisual)
    #tee.link(nvosd)

    
    if is_aarch64():
        nvosd.link(transform)
        #nvsegvisual.link(transform)
        transform.link(sink_osd)
        #transform.link(sink_seg)
        
    else:
        #nvsegvisual.link(sink_seg)
        nvosd.link(sink_osd)


    # links to file

    tee_file_pad.link(sink_queue2_pad)
    queue2.link(nvvidconvposttee)
    nvvidconvposttee.link(caps_vidconvposttee)
    caps_vidconvposttee.link(nvvenc)
    nvvenc.link(parse)
    parse.link(mux)
    mux.link(file_sink)








        
    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the src pad of the inference element
    seg_src_pad = seg.get_static_pad("src")
    if not seg_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        seg_src_pad.add_probe(Gst.PadProbeType.BUFFER, seg_src_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    
    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
