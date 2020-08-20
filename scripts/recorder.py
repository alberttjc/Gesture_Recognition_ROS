#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import datetime
import time
import rospy
import os
import glob

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from Var import Var

def opencv_version():
    v = cv2.__version__.split('.')[0]
    if v == '2':
        return 2
    elif v == '3':
        return 3
    raise Exception('opencv version can not be parsed. v={}'.format(v))


class VideoFrames:
    def __init__(self, image_topic):
        self.image_sub = rospy.Subscriber(image_topic, Image, self.callback_image, queue_size=1)
        self.bridge = CvBridge()
        self.frames = []

    def callback_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr('[ros-video-recorder][VideoFrames] Converting Image Error. ' + str(e))
            return

        self.frames.append((time.time(), cv_image))

    def get_latest(self, at_time, remove_older=True):
        fs = [x for x in self.frames if x[0] <= at_time]
        if len(fs) == 0:
            return None

        f = fs[-1]
        if remove_older:
            self.frames = self.frames[len(fs) - 1:]

        return f[1]


class VideoRecorder:
    def __init__(self, output_width, output_height, output_fps, output_format):
        self.frame_wrappers = []
        self.start_time = -1
        self.end_time = -1
        self.pub_img = None
        self.bridge = CvBridge()

        self.fps = output_fps
        self.interval = 1.0 / self.fps
        self.output_width = output_width
        self.output_height = output_height

        if opencv_version() == 2:
            self.fourcc = cv2.cv.FOURCC(*output_format)
        elif opencv_version() == 3:
            self.fourcc = cv2.VideoWriter_fourcc(*output_format)
        else:
            raise

    def add_subscription(self, subscription):
        self.frame_wrappers.append(subscription)

    def set_broadcast(self, publish_topic):
        if not publish_topic:
            self.pub_img = None
        else:
            self.pub_img = rospy.Publisher(publish_topic, Image, queue_size=1)

    def start_record(self):

        try:
            last_file_idx = sorted([int(f_name.split('/')[-1].split('.')[0]) for f_name in glob.glob(VIDEO_DIR+"*")], key=int)[-1] + 1
            #print("Last file number %d" % last_file_idx)
        except:
            print("Failed Index")
            last_file_idx = 1

        if VIDEO_DIR:
            self.video_writer = cv2.VideoWriter(VIDEO_DIR + ("%s.avi" % last_file_idx), self.fourcc, output_fps, (output_width, output_height))
            print(VIDEO_DIR + ("%s.avi" % last_file_idx))
        else:
            self.video_writer = None

        with open(LABEL_DIR +("%s.txt" % last_file_idx), "w") as label_file:

            # Print label options for user to choose from
            print(LABEL_OPTS)

            while True:
                label_choice = raw_input("Enter number to select action preformed: ")

                if label_choice == "q":
                    os.remove(VIDEO_DIR+("%s.avi" % last_file_idx))
                    os.remove(LABEL_DIR+("%s.txt" % last_file_idx))
                    self.terminate()
                    exit(0)
                try:
                    label_choice = int(label_choice)
                except:
                    continue
                break

            label_file.write(LABEL_OPTS[label_choice])

        self.start_time = time.time()
        curr_time = self.start_time + 3

        while self.end_time < 0 or curr_time <= self.end_time:
            try:
                canvas = np.zeros((self.output_height, self.output_width, 3), np.uint8)

                for frame_w in self.frame_wrappers:
                    f = frame_w.get_latest(at_time=curr_time)
                    if f is None:
                        continue

                    canvas = cv2.resize(f, (self.output_width, self.output_height))

                if self.video_writer:
                    self.video_writer.write(canvas)
                if self.pub_img:
                    try:
                        self.pub_img.publish(self.bridge.cv2_to_imgmsg(canvas, 'bgr8'))
                    except CvBridgeError as e:
                        rospy.logerr('cvbridgeerror, e={}'.format(str(e)))
                        pass

                rospy.sleep(0.01)

                if rospy.is_shutdown() and self.end_time < 0:
                    self.terminate()

                while curr_time + self.interval > time.time():
                    rospy.sleep(self.interval)

                curr_time += self.interval

                if curr_time - self.start_time > 8:
                    break

            except KeyboardInterrupt:
                self.terminate()
                continue

        if self.video_writer:
            self.video_writer.release()

    def terminate(self):
        rospy.loginfo("[ros-video-recorder] Video terminated. path={}".format(output_path))
        self.end_time = time.time()


if __name__ == '__main__':
    rospy.init_node('video_recorder', anonymous=True)

    # parameters
    output_width    =   int(rospy.get_param('~output_width', '640'))
    output_height   =   int(rospy.get_param('~output_height', '480'))
    output_fps      =   int(rospy.get_param('~output_fps', '30'))
    output_format   =   rospy.get_param('~output_format', 'x264')
    output_topic    =   rospy.get_param('~output_topic', '')
    output_path     =   rospy.get_param('~output_path', '')
    output_path     =   output_path.replace('[timestamp]', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    source_topic    =   rospy.get_param('~source', '')
    loop            =   rospy.get_param('~loop', 'False')

    VIDEO_DIR      =   (output_path + '/videos/')
    LABEL_DIR      =   (output_path + '/labels/')

    v = Var()
    LABEL_OPTS = v.get_classes()

    if not os.path.isdir(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    if not os.path.isdir(LABEL_DIR):
        os.makedirs(LABEL_DIR)

    ft = VideoRecorder(output_width, output_height, output_fps, output_format)
    vf = VideoFrames(source_topic)
    ft.add_subscription(vf)

    if output_topic:
        ft.set_broadcast(output_topic)

    # recording.
    try:
        while loop:
            ft.start_record()
    except KeyboardInterrupt:
        rospy.logerr("[ros-video-recorder] Shutting down+")

    ft.terminate()
