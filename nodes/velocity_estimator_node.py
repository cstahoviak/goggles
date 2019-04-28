#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   Apr 21, 2019
Last Edited:    Apr 21, 2019

Task: To estimate the forward and lateral body frame velocities of the sensor
platfrom given input data from a single radar (/mmWaveDataHdl/RScan topic).
The velocity estimation scheme takes the following approach:

1. Near-field targets are removed from the target list. Many of these targets
are artifacts of antenna interference at the senor origin, and are not
representative of real targets in the scene. These near-field targets also exist
in the zero-doppler bin and thus would currupt the quality of the velocity
estimate.
2. A RANSAC (or MLESAC) outlier rejection method is used to filter targets that
can be attributed to noise or dynamic targets in the environment. RANSAC
generates an inlier set of targets and a first-pass velocity estimate derived
from the inlier set.
3. Orthogonal Distance Regression (ODR) is seeded with the RANSAC velocity
estimate and generates a final estimate of the body frame linear velocity
components.

Implementation:
- The goal is for the VelocityEstimator class (node) to be model dependent
(e.g. 2D or 3D). In both cases the node is subcribed to the same topic
(/mmWaveDataHdl/RScan), and publishes a TwistWithCovarianceStamped message.
- I will need to write to separate classes (RadarDopplerModel2D and
RadarDopplerModel3D) that both define the same methods (e.g.
doppler2BodyFrameVelocity, simulateRadarDoppler, etc.) such that the
VelocityEstimator class is composed of either one of these models. This can be
thought of as designing an "implied interface" to the RadarDopplerModel subset
of classes.
- Additionally, a RadarUtilities class should be implemetned to allow access to
other functions that are model-independent.

"""

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistWithCovarianceStamped

import numpy as np
import scipy as sp
from functools import reduce
from radar_velocity_estimator.radar_utilities import RadarUtilities
from radar_velocity_estimator.radar_doppler_model_2D import RadarDopplerModel2D
import csv


class VelocityEstimator():

    def __init__(self, model, utils):
        csv_file = open('bruteForce.csv', 'a')
        self.writer = csv.writer(csv_file, delimiter=',')

        ## prescribe velocity estimator model {2D, 3D}
        self.model = model

        ## doing it this way isn't working... had to import RadarUtilities in RadarDopplerModel2D()
        self.utils = utils

        ns = rospy.get_namespace()
        rospy.loginfo("INIT: namespace = %s", ns)

        ## init subscriber
        mmwave_topic = 'radar_fwd/mmWaveDataHdl/RScan'
        # mmwave_topic = 'mmWaveDataHdl/RScan'
        self.radar_sub = rospy.Subscriber(ns + mmwave_topic, PointCloud2, self.ptcloud_cb)

        ## init publisher
        twist_topic = 'twist_estimate2'
        self.twist_pub = rospy.Publisher(ns + twist_topic, TwistWithCovarianceStamped, queue_size=10)

        ## define filtering threshold parameters - taken from velocity_estimation.m
        # self.angle_thres     = rospy.get_param('~angle_thres')
        # self.range_thres     = rospy.get_param('~range_thres')
        # self.intensity_thres = rospy.get_param('~intensity_thres')
        self.angle_thres     = 90
        self.range_thres     = 0.30
        self.intensity_thres = 5
        self.thresholds      = np.array([self.angle_thres, self.intensity_thres, self.range_thres])

        rospy.loginfo("INIT: " + ns + mmwave_topic + " angle_thres = " + str(self.angle_thres))
        rospy.loginfo("INIT: " + ns + mmwave_topic + " range_thres = " + str(self.range_thres))
        rospy.loginfo("INIT: " + ns + mmwave_topic + " intensity_thres = " + str(self.intensity_thres))

        rospy.loginfo("INIT: VelocityEstimator2D Node Initialized")

    def ptcloud_cb(self, msg):
        # rospy.loginfo("Messaged recieved on: " + rospy.get_namespace())

        pts_list = list(pc2.read_points(msg, field_names=["x", "y", "z", "intensity", "range", "doppler"]))
        pts = np.array(pts_list)

        pts[:,1] = -pts[:,1]    ## ROS standard coordinate system Y-axis is left, NED frame Y-axis is to the right
        pts[:,2] = -pts[:,2]    ## ROS standard coordinate system Z-axis is up, NED frame Z-axis is down

        ## pts.shape = (Ntargets, 6)
        if pts.shape[0] == 0:
            ## do nothing - do NOT publish a twist message: no useful velocity
            ## estimate can be derived from an empty radar message
            rospy.logwarn("WARNING: EMPTY RADAR MESSAGE")
        else:
            rospy.loginfo("New Scan\n")
            rospy.loginfo("Ntargets = %d", pts.shape[0])
            self.estimate_velocity(pts, msg)

    def estimate_velocity(self, pts, radar_msg):
        ## create target azimuth vector (in radians)
        azimuth = np.arctan(np.divide(pts[:,1],pts[:,0]))

        ## apply AIR thresholding to remove near-field targets
        data_AIR = np.column_stack((azimuth, pts[:,3], pts[:,4]))
        idx_AIR = self.utils.AIR_filtering(data_AIR, self.thresholds)
        rospy.loginfo("Ntargets_valid = %d", idx_AIR.shape[0])

        ## create vectors of valid target data
        radar_intensity = pts[idx_AIR,3]
        radar_range     = pts[idx_AIR,4]
        radar_doppler   = pts[idx_AIR,5]
        radar_azimuth   = azimuth[idx_AIR]

        if radar_doppler.shape[0] < 2:
            ## do nothing - do NOT publish a twist message: no useful velocity
            ## estimate can be derived from less than 2 targets
            rospy.logwarn("WARNING: < 2 TARGETS AFTER THRESHOLDING")
        else:
            rospy.loginfo("radar_azimuth.size = %d", radar_azimuth.size)
            rospy.loginfo("radar_azimuth.shape[0] = %d", radar_azimuth.shape[0])
            # rospy.loginfo(['{0:5.4f}'.format(i) for i in radar_azimuth])    # 'list comprehension'
            rospy.loginfo("Nbins = %d", self.utils.getNumAzimuthBins(radar_azimuth))

            ## get brute-force estimate
            model_bf, v_hat_all = self.model.getBruteForceEstimate(radar_doppler, radar_azimuth)
            rospy.loginfo(model_bf.shape)
            rospy.loginfo("model = " + str(model_bf))
            rospy.loginfo("type(model) = " + str(type(model_bf)))
            self.writer.writerow(model_bf.tolist())

            ## get RANSAC estimate + inlier set


            ## get ODR estimate

            velocity_estimate = -model_bf
            self.publish_twist_estimate(velocity_estimate, radar_msg)

    def publish_twist_estimate(self, velocity_estimate, radar_msg):

        ## create TwistStamped message
        twist_estimate = TwistWithCovarianceStamped()
        twist_estimate.header.stamp = radar_msg.header.stamp
        twist_estimate.header.frame_id = "base_link"

        twist_estimate.twist.twist.linear.x = velocity_estimate[0]
        twist_estimate.twist.twist.linear.y = velocity_estimate[1]
        twist_estimate.twist.twist.linear.z = 0

        self.twist_pub.publish(twist_estimate)


def main():
    ## anonymous=True ensures that your node has a unique name by adding random numbers to the end of NAME
    rospy.init_node('velocity_estimator_node_2D')

    csv_file = open('bruteForce.csv', 'w+')
    csv_file.close()

    # use composition to ascribe a model to the VelocityEstimator class
    velocity_estimator = VelocityEstimator(model=RadarDopplerModel2D(), utils=RadarUtilities())

    rospy.loginfo("End of main()")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        sys.exit()

if __name__ == '__main__':
    main()
