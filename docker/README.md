# To Detect object using the model

hardware requirement: GPU

1. Download and install TPOD's fork of py-faster-rcnn from github repo: 
   git clone --recursive https://github.com/junjuew/py-faster-rcnn.git 

   Follow the README to install or compile necessary packages. Note: Even if you have a caffe installed, you still want to
   compile the caffe within py-faster-rcnn dir. Such caffe will not be installed system-wide. It serves py-faster-rcnn ONLY.
2. 
        cd py-faster-rcnn/tools;
        python tpod_detect.py <input_image_path> --prototxt <prototxt_path> --weights <model_path> --labels <label_path> --output <output_image_path>
        
        example:
        python tpod_detect.py input.jpg --prototxt faster_rcnn_test.pt --weights model.caffemodel --labels labels.txt --output output.jpg


==============================================================
For Future Relase ONLY. Current docker scripts probably don't work as of 10/03/2016

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

4. Send HTTP POST request to http://localhost:<host-port>/<id> to upload image and get result

for example, use httpie:
    http --form post host:port/0 picture@Apple_iMac_Keyboard_A1242.jpeg

5. Send HTTP Get request to http://localhost:<host-port>/<query-id> to get object detection result based on id
   
      http get host:port/0
