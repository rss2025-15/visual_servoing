#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import *
# from computer_vision.sift_template import cd_sift_ransac


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.LineFollower = True

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Cone Detector Initialized")
        self.prev_time = self.get_clock().now().nanoseconds/1e9
        # self.template = cv2.imread('/root/racecar_ws/src/visual_servoing/visual_servoing/computer_vision/test_images_cone/cone_template.png')

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        # if self.LineFollower:
        ((x1, y1), (x2, y2)) = cd_color_segmentation(image, linefollower=self.LineFollower)
        # else:
            # ((x1, y1), (x2, y2)) = cd_color_segmentation(image)
        # ((x1, y1), (x2,y2)) = cd_sift_ransac(image, self.template)
        # x_pixel = (x1 + x2)//2
        x_mid = image.shape[0]//2
        if x2 < x_mid:
            x_pixel = x2
        elif x1 > x_mid:
            x_pixel = x1
        else:
            # x_pixel = x_mid
            x_pixel = (x1+x2)//2
        y_pixel = y2
        cone_pixel = ConeLocationPixel()
        cone_pixel.u = float(x_pixel)
        cone_pixel.v = float(y_pixel)
        self.cone_pub.publish(cone_pixel)

        nowtime = self.get_clock().now().nanoseconds/1e9
        fps = 1/(nowtime - self.prev_time)

        # 1) Convert the main image to HSV
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([2, 100, 100], dtype=np.uint8)
        upper_hsv = np.array([15, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
        # add box around orig image
        cv2.rectangle(image, (x1,y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, "FPS: {:.2f}".format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        combined_view = np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        debug_msg = self.bridge.cv2_to_imgmsg(combined_view, "bgr8")
        self.get_logger().info(f"{cone_pixel.u, cone_pixel.v}")
        self.debug_pub.publish(debug_msg)

        self.prev_time=nowtime

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
