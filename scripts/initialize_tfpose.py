#!/usr/bin/env python
import os
import sys
import cv2
import rospy
import rospkg
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from threading import Lock

from cv_bridge import CvBridge, CvBridgeError
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tf_pose.pafprocess as pafprocess

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, Float64, MultiArrayLayout
from Gesture_Recognition_ROS.msg import Persons, Person, BodyPartElm

from Var import Var
from PosePub import PosePub

def humans_to_msg(humans):
    persons = Persons()

    for human in humans:
        person = Person()

        for k in human.body_parts:
            body_part = human.body_parts[k]

            body_part_msg = BodyPartElm()
            body_part_msg.part_id = body_part.part_idx
            body_part_msg.x = body_part.x
            body_part_msg.y = body_part.y
            body_part_msg.confidence = body_part.score
            person.body_part.append(body_part_msg)
        persons.persons.append(person)

    return persons

def callback_image(data):
    # et = time.time()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
        return

    acquired = tf_lock.acquire(False)
    if not acquired:
        return

    try:
        humans = pose_estimator.inference(cv_image, resize_to_default=True, \
                                            upsample_size=resize_out_ratio)
        if len(humans) == 0:
            data_out.pub_null(len(humans))
        else:
            data_out.post_n_pub(humans)
    finally:
        tf_lock.release()

    msg = humans_to_msg(humans)
    msg.image_w = data.width
    msg.image_h = data.height
    msg.header = data.header

    pub_pose.publish(msg)

if __name__ == '__main__':
    """ Initialize node """
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS', anonymous=True, log_level=rospy.INFO)

    """ Initialize parameters """
    image_topic         =   rospy.get_param('~camera', '')
    model               =   rospy.get_param('~model', 'cmu')
    resolution          =   rospy.get_param('~resolution', '432x368')
    resize_out_ratio    =   float(rospy.get_param('~resize_out_ratio', '4.0'))

    tf_lock = Lock()

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided.')
        sys.exit(-1)

    try:
        w, h        =   model_wh(resolution)
        graph_path  =   get_graph_path(model)
        rospack     =   rospkg.RosPack()
        graph_path  =   os.path.join(rospack.get_path('tfpose_ros'), graph_path)

    except Exception as e:
        rospy.logerr('invalid model: %s, e=%s' % (model, e))
        sys.exit(-1)

    pose_estimator  =   TfPoseEstimator(graph_path, target_size=(w, h))
    cv_bridge       =   CvBridge()

    """ Subscirber & Publisher """
    sub_image   =   rospy.Subscriber(image_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    pub_pose    =   rospy.Publisher('~pose', Persons, queue_size=1)

    """ Initializing publisher for inference layer """
    v 			= 	Var()
    NUM_JOINTS  =   v.get_size()
    FPS         =   v.get_rate()

    rospy.loginfo("Video and Pose started")

    """ Initialize parameters """
    frame_rate      =   rospy.get_param('~frame_rate', '4')
    use_arm         =   rospy.get_param('~use_arm', 'False')
    use_angles      =   rospy.get_param('~use_angles', 'False')
    bad_data_flag   =   rospy.get_param('~bad_data_flag', 'False')
    debug           =   rospy.get_param('~debug', 'False')

    data_out = PosePub(FPS, frame_rate, use_angles, use_arm, bad_data_flag, debug)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
