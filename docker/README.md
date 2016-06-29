## Overview
This directory contains necessary files to construct a detect container using py-faster-rcnn.
The container listens on port 8080 that accept POST and GET method.
POST method receives an image, run object detection, and eventually returns a query id.
GET method is used to get the result using query id received.

## Usage

1. Name your caffe model model.caffemodel and put it under the same directory as this README file
2. Build the image by:

         docker built -t <image-name> -f Dockerfile-detect .             

3. Run the container from the image:

         nvidia-docker run -it -p <host-port>:8080 --name <container-name> <image-name>

4. Send HTTP POST request to http://localhost:<host-port>/ to upload image and get <query-id>
5. Send HTTP Get request to http://localhost:<host-port>/<query-id> to get object detection result

