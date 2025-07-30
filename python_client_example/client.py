# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import numpy as np
import cv2
import tritonclient.http as httpclient
import os


def detection_preprocessing(image: cv2.Mat) -> np.ndarray:

    data_batch = (
        np.flip(image / 255.0, axis=2)
        .astype("float32")
        .transpose(2, 0, 1)
    )
    data_batch = data_batch.reshape(1, *data_batch.shape)
    data_batch = np.ascontiguousarray(data_batch)
    return data_batch


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


def test(client):
    # Read image and create input object
    raw_image = cv2.imread("/home/data/resnet18/kitten_224.bmp")
    preprocessed_image = detection_preprocessing(raw_image)

    detection_input = httpclient.InferInput(
        "input", preprocessed_image.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)

    # Query the server
    detection_response = client.infer(
        model_name="cls", inputs=[detection_input]
    )

    # Process responses from detection model
    scores = detection_response.as_numpy("output")
    detection_postprocessing(scores)

def infer_classifier(client, image_path):
    # Read image and create input object
    raw_image = cv2.imread(image_path)
    # raw_image = cv2.resize(raw_image, (32, 32))
    preprocessed_image = detection_preprocessing(raw_image)

    detection_input = httpclient.InferInput(
        "input", preprocessed_image.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)

    # Query the server
    detection_response = client.infer(
        model_name="cls", inputs=[detection_input]
    )

    # Process responses from detection model
    scores = detection_response.as_numpy("output")
    detection_postprocessing(scores)

def infer_batch_classifier(client, image_paths):
    images = []
    for path in image_paths:
        raw_image = cv2.imread(path)
        preprocessed_image = detection_preprocessing(raw_image)
        images.append(preprocessed_image)
    batch_input = np.concatenate(images, axis=0)
    detection_input = httpclient.InferInput(
        "input", batch_input.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(batch_input, binary_data=True)
    # Query the server
    detection_response = client.infer(
        model_name="cls", inputs=[detection_input])
    # Process responses from detection model
    scores = detection_response.as_numpy("output")
    batch_size = batch_input.shape[0]
    print("\n\nbatch result")
    for i in range(batch_size):
        print("result ", i)
        detection_postprocessing(scores[np.newaxis, i, ...])

if __name__ == "__main__":
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")
    data_dir = "/home/data/resnet18"
    infer_classifier(client, os.path.join(data_dir, "kitten_224.bmp"))
    infer_classifier(client, os.path.join(data_dir,"kitten_196.bmp"))
    infer_batch_classifier(client, [os.path.join(data_dir,"kitten_224.bmp"), 
                                    os.path.join(data_dir,"robin_224.bmp")])
