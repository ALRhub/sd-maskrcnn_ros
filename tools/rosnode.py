#!/usr/bin/env python
import os
import threading
import numpy as np

import tensorflow as tf
from tensorflow.keras.backend import set_session

import rospy
import message_filters
from sensor_msgs.msg import Image

import threading

import cv2
from cv_bridge import CvBridge

from autolab_core import YamlConfig

from matplotlib import pyplot as plt

from mrcnn import visualize
from mrcnn import utils
from sd_maskrcnn.model import SDMaskRCNNModel


class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config_path = rospy.get_param('~config_path', 'cfg/benchmark.yaml')

        self.config = YamlConfig(config_path)
        self.img_width = self.config['model']['settings']['image_shape'][0]
        self.img_height = self.config['model']['settings']['image_shape'][1]


        self._visualization = rospy.get_param('~visualization', True)

        # Create model object in inference mode.
        self._model = SDMaskRCNNModel("inference", self.config["model"])
        # Load weights trained on MS-COCO

        self.depth_vis_pub = rospy.Publisher('~depth_visualization', Image, queue_size=1)
        self.rgb_vis_pub = rospy.Publisher('~rgb_visualization', Image, queue_size=1)
        
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_cb)
        self.rgb_sub   = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_cb)

        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)

        self.rgb_msg = None
        self.depth_msg = None

    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                depth_msg = self.depth_msg
                rgb_msg = self.rgb_msg
                self.rgb_msg = None
                self.depth_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if depth_msg is not None and rgb_msg is not None:
                self._image_callback(rgb_msg, depth_msg)

            rate.sleep()

    def depth_cb(self, depth_msg):
        if self._msg_lock.acquire(False):
            self.depth_msg = depth_msg
            self._msg_lock.release()

    def rgb_cb(self, rgb_msg):
        if self._msg_lock.acquire(False):
            self.rgb_msg = rgb_msg
            self._msg_lock.release()

    def _visualize(self, masks, mask_info, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(
            image, 
            mask_info['rois'], 
            np.transpose(masks, (1, 2, 0)),
            mask_info['class_ids'], 
            range(np.max(mask_info['class_ids'])),
            mask_info['scores'], 
            ax=axes,
            show_class=False
        )
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _image_callback(self, rgb_msg, depth_msg):
        rgb_image   = self._cv_bridge.imgmsg_to_cv2(rgb_msg,   'passthrough')
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        rgb_image   = cv2.resize(  rgb_image, dsize=(self.img_width, self.img_height))
        depth_image = cv2.resize(depth_image, dsize=(self.img_width, self.img_height))

        depth_image = self.depth_to_sdmaskrcnn(depth_image)

        # Run detection
        masks, mask_info = self._model.detect(depth_image)

        # Visualize results
        vis_rgb_image = self._visualize(masks, mask_info, rgb_image)
        cv_result = np.zeros(shape=vis_rgb_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_rgb_image, cv_result)
        image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
        self.rgb_vis_pub.publish(image_msg)

    def depth_to_sdmaskrcnn(self, data, min_depth=250, max_depth=1250):
        """Returns the data in image format, with scaling and conversion to uint8 types.
        Parameters
        ----------
        min_depth : float
            minimum depth value for the normalization
        max_depth : float
            maximum depth value for the normalization
        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is a set of 3 RGB values.
        """
        max_val = np.iinfo(np.uint8).max
        
        zero_px = np.where(data == 0)
        zero_px = np.c_[zero_px[0], zero_px[1]]
        depth_data = ((data - min_depth) * (float(max_val) / (max_depth - min_depth))).squeeze()
        depth_data[zero_px[:,0], zero_px[:,1]] = 0

        im_data = np.zeros([self.img_height, self.img_width, 3])
        im_data[:, :, 0] = depth_data
        im_data[:, :, 1] = depth_data
        im_data[:, :, 2] = depth_data

        return im_data.astype(np.uint8)

def main():
    rospy.init_node('mask_rcnn')

    tf_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,  allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    
    with tf.compat.v1.Session(config=tf_config) as sess:
        set_session(sess)
        with sess.as_default():
            with sess.graph.as_default():

                node = MaskRCNNNode()
                node.run()

if __name__ == '__main__':
    main()
