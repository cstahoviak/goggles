"""
Author:         Carl Stahoviak
Date Created:   Apr 22, 2019
Last Edited:    Apr 22, 2019

Description:

"""

import rospy
import numpy as np
from functools import reduce

class RadarUtilities():

    def __init__(self):
        pass

    def AIR_filtering(self, data_AIR, thresholds):
        ## unpack data (into row vectors)
        radar_azimuth   = data_AIR[:,0]     # [rad]
        radar_intensity = data_AIR[:,1]     # [dB]
        radar_range     = data_AIR[:,2]     # [m]

        azimuth_thres   = thresholds[0];
        intensity_thres = thresholds[1];
        range_thres     = thresholds[2];

        ## Indexing in Python example
        ## print("Values bigger than 10 =", x[x>10])
        ## print("Their indices are ", np.nonzero(x > 10))
        idx_angle = np.nonzero(np.absolute(np.rad2deg(radar_azimuth)) < azimuth_thres);
        idx_intensity = np.nonzero(radar_intensity > intensity_thres);
        idx_range = np.nonzero(radar_range > range_thres);

        ## combine filters
        idx_AIR = reduce(np.intersect1d, (idx_angle, idx_range, idx_intensity))
        return idx_AIR

    def getNumAzimuthBins(self, radar_azimuth):
        bin_thres = 0.009;      # [rad] - empirically determined

        azimuth_bins = np.unique(radar_azimuth);

        bins = [];
        current_bin = azimuth_bins[0];
        begin_idx = 0;

        for i in range(azimuth_bins.shape[0]):
            if np.absolute(current_bin - azimuth_bins[i]) > bin_thres:
                ## update location of last angle to be averaged
                end_idx = i-1;

                ## add single averaged value to table
                # print(begin_idx, end_idx)
                azimuth_bin = np.mean(azimuth_bins[begin_idx:end_idx]);
                bins.append(azimuth_bin)

                ## set new beginning index
                begin_idx = i;

                ## set new current bin
                current_bin = azimuth_bins[i];

            if i == azimuth_bins.shape[0]-1:
                ## update location of last angle to be averaged
                end_idx = i;

                ## add single averaged value to table
                azimuth_bin = np.mean(azimuth_bins[begin_idx:end_idx]);
                bins.append(azimuth_bin)

        bins = np.array(bins)
        numAzimuthBins = bins.size

        return numAzimuthBins
