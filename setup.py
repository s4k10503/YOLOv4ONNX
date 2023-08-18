import os
import urllib.request
import shutil


def download_model(model_url: str, model_name: str) -> None:
    model_folder = "./model"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, model_name)

    with urllib.request.urlopen(model_url) as response, open(model_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    print(f"ONNX model {model_name} saved to {model_path}")


if __name__ == "__main__":
    download_model(
        "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx?download=", "yolov4.onnx")
