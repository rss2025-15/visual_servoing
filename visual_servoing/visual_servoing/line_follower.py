#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.time import Time
import time
import math

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header

class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(ConeLocation, "/relative_cone", 
            self.relative_cone_callback, 1)

        self.parking_distance = .7 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.wheelbase = .46
        self.park_thres = 0.1
        self.max_steer = 0.78
        
        self.min_turn_radius = self.wheelbase/math.tan(self.max_steer)

        self.speed = 1.

        self.no_cone()

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        lookahead = np.linalg.norm([self.relative_x, self.relative_y])
        # gain = 1.
        self.get_logger().info(f"lookahead {lookahead}")

        # plan is to follow a circular trajectory to go to the cone (which will be smooth)
        # if the computed radius is smaller than the minimum turning radius of the robot we should also go backwards
        # (at least to the point where the turning radius is the minimum turning radius (tangency of two circles))
        # and then it should go forwards

        turn_radius = lookahead / (2*math.sin(math.atan2(self.relative_y,self.relative_x)))

        if self.relative_x < 0:
            self.get_logger().info('STOP')
            self.stop_cmd()

        if lookahead > self.parking_distance+self.park_thres and abs(turn_radius) >= self.min_turn_radius:
        # if lookahead > self.parking_distance:
            # forward pure pursuit (intersection with straight line towards cone? maybe change to circle?)
            # self.drive_cmd(gain*np.arctan(lookahead*self.wheelbase/(2*self.relative_y)))

            steer_angle = math.atan(self.wheelbase/turn_radius)
            speed = 1.0-math.exp(-(lookahead-self.parking_distance))
            if speed < 0.5:
                speed = 0.0
            self.drive_cmd(steer_angle, 1.0-math.exp(-(lookahead-self.parking_distance)))
            self.get_logger().info('FORWARD, STEERING {steer_angle}')
                    
        elif lookahead < self.parking_distance-self.park_thres or abs(turn_radius) < self.min_turn_radius:
            # go back and turn
            if self.relative_y > 0:
                # cone is to the left, go back right
                self.drive_cmd(-self.max_steer, -1.0)
                self.get_logger().info('FULL BACK RIGHT')
            else:
                # cone right, go back left
                self.drive_cmd(self.max_steer, -1.0)
                self.get_logger().info('FULL BACK LEFT')
        else:
            self.get_logger().info('STOP')
            self.stop_cmd()
        
        self.error_publisher()
    
    def no_cone(self):
        self.drive_cmd(2.0, 0.5)

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)
        
        self.error_pub.publish(error_msg)

    def drive_cmd(self, steer, speed = 1.0):
        drive_cmd_drive = AckermannDriveStamped()
        drive_cmd_drive.drive.speed = speed
        drive_cmd_drive.drive.steering_angle = steer
        drive_cmd_drive.drive.steering_angle_velocity = 0.0
        drive_cmd_drive.drive.acceleration = 0.0
        drive_cmd_drive.drive.jerk = 0.0
        drive_cmd_drive.header.stamp = self.get_clock().now().to_msg()
        self.drive_pub.publish(drive_cmd_drive)
    
    def stop_cmd(self):
        stop_cmd_drive = AckermannDriveStamped()
        stop_cmd_drive.drive.speed = 0.0
        stop_cmd_drive.drive.steering_angle = 0.0
        stop_cmd_drive.drive.steering_angle_velocity = 0.0
        stop_cmd_drive.drive.acceleration = 0.0
        stop_cmd_drive.drive.jerk = 0.0
        stop_cmd_drive.header.stamp = self.get_clock().now().to_msg()
        self.drive_pub.publish(stop_cmd_drive)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()