# Lab 8: Vision Lab

## I. Learning Goals

- Use of v4l2 to access and control camera device.
- Camera calibration with OpenCV
- Distance measurement with object on the ground
- Training and Deployment of an object detection with TensorRT

## II. Use cameras on Linux system

As we have discussed in the lecture, in Linux cameras are mounted as file systems. Connect the Realsense camera you have received to Jetson and check if you can see three outputs with the `ls /dev/video*` command. Use OpenCV to capture RGB frames from `/dev/video2`.

During racing, we want to receive camera frames as fast as possible. Realsense camera only operates at 60Hz with a 960x540 resolution. Please modify your code so that we have receiving 960x540 images at 60Hz.

Then write a ROS2 node that captures RGB frame from the Realsense camera and publish it onto `/rgb_img` channel using `sensor_msgs/Image` message type.

## III. Camera Calibration & Distance Measurement

In this part, you will develop a distance measurement algorithm with the assumption that the camera frame axes are overlapping the car frame axes with a height H on the Z axis of car frame.

First, we need to get the camera intrinsic matrix from a camera calibration. The images needed for such a calibration is provides in the 'calibration' folder. Please use `findChessboardCorners` and `calibrateCamera` to get the intrinsic matrix.

Second, the height of the camera can be difficult to measure precisely. We need corresponding point in the car frame to calculate this height. In the 'distance' folder, there are two images with measured x_car distance and two images with unknown distances. 

Please calculate the mounting height. Then write a function that takes in image pixel coordinates and outputs the distance in x_car and y_car coordinate. And for the two images with unknown distances, calculated the distances from the camera to the cone. 

Note 1: All distances are with respect to the lower right corner of the nearest red cone.
![cone_point](imgs/cone_point.png)

Note 2: x, y coordinates are different from row and column in an image.

Note 3: All images are in 960x540 resolution.

<!-- ![](https://latex.codecogs.com/svg.latex?b) -->

## IV. Lane Detection

In this part, you will develop a lane detection algorithm with classical computer vision. In the 'lane' folder, there is one image with yellow lane marks. Your task is to detect the lane maskers as good as possible, using HSV color space, blob detection, erode or any other functions.

Please write a function that take in an image and draw green edges around or green overlaps on the marks you detect.

## V. Object Detection Network Training & Deployment

In this part, you will first train an object detection nerual network and deploy it onto the F110 car. 

### Nerual Network Training

#### Upload to Google Colab

For the network training, we have provided labeled dataset. If you are new to neural network training, we have provided a simple YOLO training and testing notebook written in Pytorch. You can use it to train on Google Colab with the free GPU resource. If you are new to Pytorch, watch this [tutorial](https://www.youtube.com/watch?v=Jy4wM2X21u0) as it goes through some key components in a training pipeline.

You can create an ese_615 folder in the 'My Drive' in your Google Drive. Upload the notebook and unzipped dataset folder. Open the note book with Google Colab and make sure you are connected to a GPU instance. The notebook has everything you need to for training and testing. For some of post-processing functions, you can just borrow them when you do you deployment.

#### TODOs

There are only a few positions where marked 'TODO'. First is the network layer channels. Play around with different channel configurations. The first input channel should be 3, because we are feeding it with RGB image. The final output layers should be 5, because we have x, y, w, h and confidence. We don't have class, because we are only detecting F110 cars. Make sure the output channel of the upper layer needs to match the batchnorm size and the input channel of the lower layer. Usually the channels are set as multiplies of 2.

The next 'TODO' are some hyperparameters: batchsize, learning rate and epoch. Large batchsizes make you learn faster, but lower batchsizes give more randomness. Large learning rate make you learn faster but may be unstable (producing NaNs) and bounce around the optimal point. Epoch is just how many times you run with all the dataset. Play around with this values and get a sense of what's suitable. Since our detection task is simple and the dataset is not diverse. You should be able to drop the training error below 1000 quite easily and you won't need to pay for a fast GPU to complete this lab. We are not grading on the accuracy, as long as it can detect somewhat.

If you are familiar with network training, you can choose any network architecture you like and train it with our dataset. Maybe try some newly published networks that are transformer based? You need to make sure that it can be successfully deployed with TensorRT. Not all layers are supported by TensorRT. 

### TensorRT Deployment

#### Install Pytorch
If your Jetson doesn't have Pytorch, follow [this](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) to install Pytorch.
You can install matplotlib with `sudo apt-get install python3-matplotlib`.

#### Install pycuda and set up TensorRT
If you Jetson doesn't have pycuda, follow [this](https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/) to set up TensorRT and install pycuda.

After you are done with training the network, convert it into a TensorRT engine on the Jetson NX. To do this, first convert the model from PyTorch into ONNX and then to TensorRT. Then write a function takes in an image, preprocess it, run with the model, post-process and return the results. Tutorial and sample code can be found [here](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/).

Trying converting the the engine using FP32 and FP16 mode and compare the speed difference.

## VI. Combining the Functions

In the final part, we will write another ROS node that subscribes to `/rgb_img` channel for an image. Run the lane detection and object detection on the image. Then use the center point of the botton bounding box edge from object detection to calcuate the distance to the object. Please note that the image resolution should be 960x540 and your detection resolution is smaller.

Publish the result image with lane detection and object detection indicators to the `/det_img` channel. Publish the distance to detections to the `/det_distance` channel.

## VII. Deliverables and Submission

Please follow the `submission.md` in the Github repo.

## VIII: Grading Rubric

- Accessing Image: **10** Points
- Distance Measurement: **15** Points
- Lane Detection: **20** Points
- Network Training: **15** Points
- TensorRT Deployment: **30** Points
- Integration: **10** Points
