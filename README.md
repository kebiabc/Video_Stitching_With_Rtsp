# Video Stitching and Streaming System with Iron Slab Detection

## Overview
This project implements an image stitching and streaming system that captures images from multiple cameras, stitches them into a panoramic view, and streams the result in real-time via RTSP. Additionally, it listens for target coordinates via ZeroMQ and marks the target position dynamically on the stitched image.

## Features
- **Multi-Camera Image Stitching**: Captures images from multiple cameras and combines them into a single panoramic view.
- **Real-Time Target Marking**: Receives target coordinates from a ZeroMQ subscriber and marks them on the stitched image.
- **RTSP Streaming**: Encodes the final stitched image and streams it using `libx264` over RTSP.
- **Threaded Processing**: Utilizes multithreading to achieve efficient image processing and streaming.

## Dependencies
Ensure you have the following dependencies installed:

- OpenCV
- ZeroMQ (`zmq`)
- FFmpeg (`libavformat`, `libavcodec`, `libswscale`, `libavutil`)
- C++ Standard Library (C++11 or newer)

## Installation
```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repository>

# Install dependencies (example for Ubuntu)
sudo apt update && sudo apt install -y libopencv-dev libzmq3-dev libavformat-dev libavcodec-dev libswscale-dev libavutil-dev

# Compile the project
mkdir build && cd build
cmake ..
make
```

## Usage
Run the application with:
```bash
./image_stitching
```

## Configuration
### ZeroMQ Target Input
The system listens for target coordinates from a ZeroMQ publisher on `tcp://localhost:5555`. The expected message format is:
```
<camera_id>,<x>,<y>
```
where:
- `<camera_id>`: The ID of the camera where the target is detected.
- `<x>, <y>`: The pixel coordinates in the corresponding camera frame.

### RTSP Streaming
The stitched image is streamed to:
```
rtsp://192.168.1.81:8554/live
```
This can be adjusted in the `PushFrame` function inside `App.cpp`.

## Logging
The system prints processing times for:
- Image capture
- Image stitching
- Frame pushing
- Total processing time per frame

Additionally, received target coordinates and the marked position in the stitched image are printed.

## Contributing
Feel free to fork and submit pull requests. Any contributions to improve image stitching accuracy, streaming performance, or feature enhancements are welcome!

## License
This project is licensed under the MIT License.

