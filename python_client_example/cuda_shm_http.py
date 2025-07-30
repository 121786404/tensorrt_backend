#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys
from builtins import range

import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient import utils

import cv2


FLAGS = None


def detection_postprocessing(scores):
    data = scores
    # print(type(data))
    # print(data.shape)

    from imagenet_labels import labels
    from topk import topk
    vals, idxs = topk(data, 5, axis=1)
    idx0 = idxs[0]
    val0 = vals[0]
    print(
        "------------------------------Python inference result------------------------------"
    )
    for i, (val, idx) in enumerate(zip(val0, idx0)):
        print(f"Top {i+1}:   {val}  {labels[idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )

    FLAGS = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # To make sure no shared memory regions are registered with the
    # server.
    triton_client.unregister_system_shared_memory()
    triton_client.unregister_cuda_shared_memory()

    # We use a simple model that takes 2 input tensors of 16 integers
    # each and returns 2 output tensors of 16 integers each. One
    # output tensor is the element-wise sum of the inputs and one
    # output is the element-wise difference.
    model_name = "cls"
    model_version = ""

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.

    raw_image = cv2.imread("/home/zhichao.yang/baidu/resnet18/kitten_224.bmp")

    input0_data = (
        np.flip(raw_image / 255.0, axis=2)
        .astype("float32")
        .transpose(2, 0, 1)
    )

    input0_data = input0_data.reshape(1, *input0_data.shape)
    input0_data = np.ascontiguousarray(input0_data)
    input0_data = input0_data.reshape(1, 3, 224, 224)

    input_byte_size = input0_data.size * input0_data.itemsize
    output_byte_size = 1000 * 1 * 4

    print(f"data[0]{input0_data[0][0][0][0]}")

    # Create Output0 and Output1 in Shared Memory and store shared memory handles
    shm_op0_handle = cudashm.create_shared_memory_region(
        "output0_data", output_byte_size, 0
    )

    # Register Output0 and Output1 shared memory with Triton Server
    triton_client.register_cuda_shared_memory(
        "output0_data", cudashm.get_raw_handle(shm_op0_handle), 0, output_byte_size
    )

    # Create Input0 and Input1 in Shared Memory and store shared memory handles
    shm_ip0_handle = cudashm.create_shared_memory_region(
        "input0_data", input_byte_size, 0
    )

    # Put input data values into shared memory
    cudashm.set_shared_memory_region(shm_ip0_handle, [input0_data])

    # Register Input0 and Input1 shared memory with Triton Server
    triton_client.register_cuda_shared_memory(
        "input0_data", cudashm.get_raw_handle(shm_ip0_handle), 0, input_byte_size
    )

    # Set the parameters to use data from shared memory
    inputs = []
    inputs.append(httpclient.InferInput("input", [1, 3, 224, 224], "FP32"))
    inputs[0].set_shared_memory("input0_data", input_byte_size)


    outputs = []
    outputs.append(httpclient.InferRequestedOutput("output", binary_data=True))
    outputs[0].set_shared_memory("output0_data", output_byte_size)


    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    # Read results from the shared memory.
    output0 = results.get_output("output")
    if output0 is not None:
        output0_data = cudashm.get_contents_as_numpy(
            shm_op0_handle,
            utils.triton_to_np_dtype(output0["datatype"]),
            output0["shape"],
        )
    else:
        print("OUTPUT0 is missing in the response.")
        sys.exit(1)

    #print(f"output0[0][0]:{output0_data[0][0]}")

    detection_postprocessing(output0_data)

    print(triton_client.get_cuda_shared_memory_status())
    triton_client.unregister_cuda_shared_memory()
    cudashm.destroy_shared_memory_region(shm_ip0_handle)
    cudashm.destroy_shared_memory_region(shm_op0_handle)
    assert len(cudashm.allocated_shared_memory_regions()) == 0

    print("PASS: cuda shared memory")