# YOLOv4 ONNX

## Overview

This repository contains code to perform object detection on images and videos using the YOLOv4 model provided in ONNX format.

## Features

- Fast inference using ONNX Runtime
- Support for both images and videos
- GPU support

## Setup

Please run the setup.py to download the required files.
(DL the model of [ONNX Model Zoo (YOLOv4).](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4))

## Usage

Run main.py to launch the application.

## Docker Support

Running this application using Docker is also possible and highly recommended for ensuring the environment is set up correctly.

### Prerequisites

- Docker installed
- nvidia-docker if you are planning to use GPU support

### Build the Docker Image

Run the following command to build the Docker image. This will install all the necessary packages and set up the environment.

```bash
docker build -t yolov4_onnx .
```

### Run the Docker Container

To run the application inside a Docker container, execute the following command. This also mounts the host directory to the container, enabling file access for the GUI.

```bash
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/on/host:/path/in/container --device=/dev/video0:/dev/video0 yolov4_onnx
```

Note: Replace /path/on/host and /path/in/container with the appropriate paths to mount the directory from your host machine into the container.
Security Note: Before running the container, make sure to execute xhost + on the host machine to allow connections to the X server.

```bash
xhost +
```

```bash
xhost -
```

## Contributing

Feel free to fork the repository and submit pull requests for any enhancements or bug fixes.

## License

The code in this repository is licensed under the MIT License. See the LICENSE file for details.  
However, please take care to follow the rules and regulations of the quotation for the portions quoted or modified from [ONNX Model Zoo (YOLOv4).](https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov4)
