
# Gesture_Recognition_ROS
This repo was restructured from several repos and integrated with Python 2.7 and Tensorflow 1.14 for the purposes of implementing in in ROS Melodic.

------
## Introduction
*The **pipeline** of this work is:*   
 - [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose);   
 - [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)
 - [Gesture_Recognition](https://github.com/NVIDIA-AI-IOT/Gesture-Recognition)

------
## Dependencies
These were the dependencies tested with this code:
 - python = 2.7 (ROS)
 - tensorflow = 1.14  
 - pytorch = 1.4

**Please ensure you satisfy all dependencies from the repos above**

------
## Implementation

1. Change the image topic for your subscriber in tfpose.launch
```
  <arg name="camera_topic" default="/image_topic" />
```

2. Change the path to the checkpoint file in inference.launch
```
  <arg name="ckpt_path" default = "/path/to/lstm623.ckpt" />
```

3. Launch tfpose.launch first and wait for it to initalize before launching the inference layer

```
$ roslaunch Gesture_Recognition_ROS tfpose.launch
$ roslaunch Gesture_Recognition_ROS inference.launch
```
