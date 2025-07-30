import math
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
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

    detection_input = grpcclient.InferInput(
        "input", preprocessed_image.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(preprocessed_image)

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

    detection_input = grpcclient.InferInput(
        "input", preprocessed_image.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(preprocessed_image)

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
    detection_input = grpcclient.InferInput(
        "input", batch_input.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(batch_input)
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
    client = grpcclient.InferenceServerClient(url="localhost:8001")
    data_dir = "/home/tis_ixrt_backend_data/resnet18"
    infer_classifier(client, os.path.join(data_dir, "kitten_224.bmp"))
    infer_classifier(client, os.path.join(data_dir,"kitten_196.bmp"))
    infer_batch_classifier(client, [os.path.join(data_dir,"kitten_224.bmp"), 
                                    os.path.join(data_dir,"robin_224.bmp")])
