#! /usr/bin/env python
import rospy
import numpy as np
import scipy as sp

from radar_velocity_estimator.radar_utilities import RadarUtilities

def main():
    radar_azimuth_bins = np.genfromtxt('1642_azimuth_bins.csv', delimiter=',')
    # print(radar_azimuth_bins)

    utils = RadarUtilities()

    numAzimuthBins = utils.getNumAzimuthBins( radar_azimuth_bins )
    print(numAzimuthBins)


if __name__ == '__main__':
    main()
