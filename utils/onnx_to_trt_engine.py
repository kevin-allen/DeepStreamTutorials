import os
import random
import sys
import numpy as np

# This import causes pycuda to automatically manage CUDA context creation and cleanup.

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.

import tensorrt as trt


sys.path.insert(1, os.path.join(sys.path[0], ".."))
#import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network( 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = 1 * 1 << 30
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_engine(network, config)




def main():
    # Set the data path to the directory that contains the trained models and test images for inference.

    onnx_model_file="/home/kevin/repo/DeepStreamTutorials/models/UNet.onnx"

    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)

    # Print information about the engine
    inspector = engine.create_engine_inspector()
    print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    
    
    
    engine_file="/home/kevin/repo/DeepStreamTutorials/models/serialized_UNet_engine_batch2.trt"
    with open(engine_file, "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":
    main()
