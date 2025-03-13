#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.time import Time
import time

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

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

        self.parking_distance = .75 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.wheelbase = .46

        self.speed = 1.

        self.no_cone()
    

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        lookahead = np.linalg.norm([self.relative_x, self.relative_y])
        gain = lookahead *0.1
        self.get_logger().info(f"lookahead {lookahead}")

        if lookahead > self.parking_distance:

            steer = np.arctan(lookahead*self.wheelbase/(2*self.relative_y))
            self.speed= 1.

        else:
            steer = 0.
            self.speed = 0.
            
        drive_cmd.drive.steering_angle = steer*gain
        drive_cmd.drive.steering_angle_velocity = 0.0

        drive_cmd.drive.speed = self.speed
        drive_cmd.drive.acceleration = 0.0
        drive_cmd.drive.jerk = 0.0

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()
    
    def no_cone(self):
        drive_cmd = AckermannDriveStamped()
        drive_cmd.speed = 0.5
        drive_cmd.steering_angle = 2.0
        drive_cmd.drive.steering_angle_velocity = 0.0
        drive_cmd.drive.acceleration = 0.0
        drive_cmd.drive.jerk = 0.0

        self.drive_pub.publish(drive_cmd)

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)


        #################################
        
        self.error_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()

if __name__ == '__main__':
    main()