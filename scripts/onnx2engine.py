import tensorrt as trt
from tensorrt import Dims
import os
dir_path="/home/data/resnet18/"
onnx_name = 'resnet18-all-dynamic.onnx'
engine_name = "resnet18_dynamic.engine"

TRT_LOGGER = trt.Logger()
builder = trt.Builder(TRT_LOGGER)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)
parser.parse_from_file(os.path.join(dir_path, onnx_name))

profile = builder.create_optimization_profile()
profile.set_shape("input", Dims([1,3,112,112]), Dims([2,3,224,224]), Dims([16,3,448,448]))
config.add_optimization_profile(profile)
config.set_flag(trt.BuilderFlag.FP16)

engine_file_path = os.path.join(dir_path, engine_name)
plan = builder.build_serialized_network(network, config)
with open(engine_file_path, "wb") as fh:
    fh.write(plan)
