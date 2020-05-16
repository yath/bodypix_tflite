import tensorflow as tf
import sys
import os
sys.path.insert(0, (os.path.dirname(__file__) or ".")+"/deps/simple_bodypix_python")
import utils

tfjs_filename, graphdef_filename, width, height, pu_type, output_tensor_suffix, output_filename = \
        sys.argv[1:]

graph = utils.load_graph_model(tfjs_filename)

input_tensors = utils.get_input_nodes(graph)
assert len(input_tensors) == 1

input_tensor_name = input_tensors[0].name
input_tensor_shape = input_tensors[0].shape
assert input_tensor_shape[1] is None
input_tensor_shape[1] = int(width)
assert input_tensor_shape[2] is None
input_tensor_shape[2] = int(height)

print("Selected input tensor {} with shape {}".format(input_tensor_name, input_tensor_shape))

output_tensors = utils.get_output_nodes(graph)
output_tensor = [t for t in output_tensors if t.name.endswith(output_tensor_suffix)]
assert len(output_tensor) == 1
output_tensor = output_tensor[0]

print("Selected output tensor {} with shape {}".format(output_tensor.name, output_tensor.shape))

print("Converting...")
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graphdef_filename,
        input_arrays=[input_tensor_name],
        input_shapes={input_tensor_name: input_tensor_shape},
        output_arrays=[output_tensor.name],
)
converter.allow_custom_ops=True
converter.optimizations=[tf.lite.Optimize.DEFAULT]
if pu_type == "gpu":
    converter.target_spec.supported_types = [tf.float16]
else:
    assert pu_type == "cpu"
tflite_model = converter.convert()
open(output_filename, "wb").write(tflite_model)
print("Wrote {:,} bytes to {}".format(len(tflite_model), output_filename))
