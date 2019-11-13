#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   Nov 07, 2019
Last Edited:    Nov 07, 2019

Task: To remove additional points from the /mmWaveDataHdl/RScanInliers topic for
use with the Octomap mapping package

"""

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

class RScanFilter():

    def __init__(self):

        ns = rospy.get_namespace()
        rospy.loginfo("INIT: namespace = %s", ns)
        if ns =='/':
            ## empty namespace
            ns = ns[1:]

        ## read input parameters
        input_pcl_topic = rospy.get_param('~input_pcl')
        self.z_thres_ = rospy.get_param('~z_thres')
        self.intensity_thres_ = rospy.get_param('~intensity_thres')

        self.pcl_sub = rospy.Subscriber(ns + input_pcl_topic, PointCloud2, self.ptcloud_cb)
        rospy.loginfo("INIT: RScanFilter Node subcribed to:\t" + ns + input_pcl_topic)

        rscan_filtered_topic = input_pcl_topic + '/filtered'
        self.pc_pub = rospy.Publisher(ns + rscan_filtered_topic, PointCloud2, queue_size=10)
        rospy.loginfo("INIT: RScanFilter Node publishing on:\t" + ns + rscan_filtered_topic)

        rospy.loginfo("INIT: RScanFilter Node Initialized")

    def ptcloud_cb(self, radar_msg):
        pts_list = list(pc2.read_points(radar_msg, field_names=["x", "y", "z", "intensity", "range", "doppler"]))
        pts = np.array(pts_list, dtype=np.float32)

        ## pts.shape = (Ntargets, 6)
        if pts.shape[0] == 0:
            rospy.logwarn("filter_rscan: EMPTY RADAR MESSAGE")
        else:
            self.filter_rscan(pts, radar_msg)

    def filter_rscan(self, pts, radar_msg):

        ## remove points outside of specified z threshold
        radar_z = pts[:,2]
        radar_intensity = pts[:,3]

        idx_z = np.nonzero( np.abs(radar_z) < self.z_thres_ )
        idx_int = np.nonzero( radar_intensity > self.intensity_thres_ )
        idx = reduce(np.intersect1d,(idx_z,idx_int))
        pts_filtered = pts[idx,:]

        self.publish_filtered_rscan(pts_filtered, radar_msg)

    def publish_filtered_rscan(self, pts_filtered, radar_msg):

        rscan_filtered = PointCloud2()

        rscan_filtered.header = radar_msg.header

        rscan_filtered.height = 1
        rscan_filtered.width = pts_filtered.shape[0]
        rscan_filtered.is_bigendian = False
        rscan_filtered.point_step = 24
        rscan_filtered.row_step = rscan_filtered.width * rscan_filtered.point_step
        rscan_filtered.is_dense = True

        ## this is not being done correctly...
        rscan_filtered.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('range', 16, PointField.FLOAT32, 1),
            PointField('doppler', 20, PointField.FLOAT32, 1),
            ]

        data = np.reshape(pts_filtered,(pts_filtered.shape[0]*pts_filtered.shape[1],))
        rscan_filtered.data = data.tostring()
        self.pc_pub.publish(rscan_filtered)

def main():
    ## anonymous=True ensures that your node has a unique name by adding random numbers to the end of NAME
    rospy.init_node('rscan_filter_node')
    rscan_filter = RScanFilter()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        sys.exit()

if __name__ == '__main__':
    main()
