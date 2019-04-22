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
2. a RANSAC (or MLESAC) outlier rejection method is used to filter targets that
can be attributed to noise of dynamic targets in the environment. RANSAC
generates an inlier set of targets and first-pass velocity estimate derived from
the inlier set.
3. Orthogonal Distance Regression (ODR) is seeded with the RANSAC velocity
estimate and generates a final estimate of the body frame linear velocity
components.
"""

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistStamped, TwistWithCovarianceStamped

import numpy as np
from functools import reduce

class VelocityEstimator2D():

    def __init__(self):
        ns = rospy.get_namespace()
        rospy.loginfo("INIT: namespace = %s", ns)

        ## init subscriber
        mmwave_topic = 'mmWaveDataHdl/RScan'
        self.radar_sub = rospy.Subscriber(ns + mmwave_topic, PointCloud2, self.ptcloud_cb)

        ## init publisher
        twist_topic = 'twist_estimate'
        self.twist_pub = rospy.Publisher(ns + twist_topic, TwistWithCovarianceStamped, queue_size=10)

        ## define filtering threshold parameters - taken from velocity_estimation.m
        self.angle_thres     = rospy.get_param('~angle_thres')
        self.range_thres     = rospy.get_param('~range_thres')
        self.intensity_thres = rospy.get_param('~intensity_thres')

        rospy.loginfo("INIT: " + ns + mmwave_topic + " angle_thres = " + str(self.angle_thres))
        rospy.loginfo("INIT: " + ns + mmwave_topic + " range_thres = " + str(self.range_thres))
        rospy.loginfo("INIT: " + ns + mmwave_topic + " intensity_thres = " + str(self.intensity_thres))

        rospy.loginfo("INIT: VelocityEstimator2D Node Initialized")

    def ptcloud_cb(self, msg):
        # rospy.loginfo("Messaged recieved on: " + rospy.get_namespace())

        pts_list = list(pc2.read_points(msg, field_names=["x", "y", "z", "intensity", "range", "doppler"]))
        pts = np.array(pts_list)

        if pts.shape[0] == 0:
            ## do nothing - do NOT publish a twist message: no useful velocity
            ## estimate can be derived from an empty radar message
            rospy.logwarn("WARNING: EMPTY RADAR MESSAGE")
        else:
            self.estimate_velocity(pts, msg)

    def estimate_velocity(self, pts, radar_msg):
        # rospy.loginfo("pts.shape = " + str(pts.shape))

        ## create target azimuth vector (in radians)
        radar_azimuth = np.arctan(np.divide(pts[:,1],pts[:,0]))

        ## Indexing in Python example
        ## print("Values bigger than 10 =", x[x>10])
        ## print("Their indices are ", np.nonzero(x > 10))
        idx_angle = np.nonzero(np.absolute(np.rad2deg(radar_azimuth)) < self.angle_thres);
        idx_range = np.nonzero(pts[:,4] > self.range_thres);
        idx_intensity = np.nonzero(pts[:,3] > self.intensity_thres);

        ## apply (angle,range,intensity) filtering to doppler data
        idx_ARI = reduce(np.intersect1d, (idx_angle, idx_range, idx_intensity))

        radar_doppler = pts[:,5]
        doppler_filtered = radar_doppler[idx_ARI];

        if doppler_filtered.size < 2:
            ## do nothing - do NOT publish a twist message: no useful velocity
            ## estimate can be derived from less than 2 targets
            rospy.logwarn("WARNING: < 2 TARGETS AFTER THRESHOLDING")
        else:
            ## get RANSAC estimate + inlier set


            ## get ODR estimate


            self.publish_twist_estimate(velocity_estimate, radar_msg)

    def publish_twist_estimate(self, velocity_estimate, radar_msg):

        ## create TwistStamped message
        twist_estimate = TwistWithCovarianceStamped()
        twist_estimate.header.stamp = radar_msg.header.stamp
        twist_estimate.header.frame_id = "base_link"

        twist_estimate.twist.linear.x = velocity_estimate.x
        twist_estimate.twist.linear.y = velocity_estimate.y
        twist_estimate.twist.linear.z = 0

        self.twist_pub.publish(twist_estimate)


def main():
    ## anonymous=True ensures that your node has a unique name by adding random numbers to the end of NAME
    rospy.init_node('velocity_estimator_node_2D')
    velocity_estimator = VelocityEstimator2D()

    rospy.loginfo("End of main()")

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        sys.exit()

if __name__ == '__main__':
    main()
