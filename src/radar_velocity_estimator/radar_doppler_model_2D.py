"""
Author:         Carl Stahoviak
Date Created:   Apr 22, 2019
Last Edited:    Apr 22, 2019

Description:

"""

import rospy
import numpy as np
import scipy as sp

from radar_velocity_estimator.radar_utilities import RadarUtilities

class RadarDopplerModel2D():

    def __init__(self):
        self.utils = RadarUtilities()

    # defined for RANSAC
    def fit(self, data):
        pass

        # radar_azimuth = data[:,1]
        # radar_doppler = data[:,2]
        #
        # model = doppler2BodyFrameVelocity2D(radar_doppler, radar_azimuth)
        # return model

    # defined for RANSAC
    def get_error(self, data, model):
        pass

        # ## number of targets in scan
        # Ntargets = size(data,1);
        # distances = zeros(Ntargets,1);
        #
        # radar_angle   = data(:,1);    % [rad]
        # radar_doppler = data(:,2);    % [m/s]
        #
        # ## do NOT corrupt measurements with noise
        # eps = zeros(Ntargets,1);
        # delta = zeros(Ntargets,1);
        #
        # ## radar doppler generative model
        # doppler_predicted = simulateRadarDoppler2D(model, ...
        #     radar_angle, eps, delta);
        #
        # for i=1:Ntargets
        #     error(i) = sqrt((doppler_predicted(i) - radar_doppler(i))^2);
        # end

    # inverse measurement model: measurements->model
    def doppler2BodyFrameVelocity(self, radar_doppler, radar_azimuth):
        pass

        numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)

        if numAzimuthBins > 1:
           ## solve uniquely-determined problem for pair of targets (i,j)
            M = np.array([[np.cos(radar_azimuth[0]), np.sin(radar_azimuth[0])], \
                          [np.cos(radar_azimuth[1]), np.sin(radar_azimuth[1])]])

            b = np.array([[radar_doppler[0], radar_doppler[1]]])
            model = np.linalg.solve(M,np.transpose(b))
        else:
            model = float('nan')*np.ones((2,1))

        return np.transpose(model)

    # measurement generative (forward) model: model->measurements
    def simulateRadarDoppler(self, data, model):
        pass

    def getBruteForceEstimate(self, radar_doppler, radar_azimuth):

        ## number of targets in scan
        Ntargets = radar_doppler.shape[0]
        iter = (Ntargets-1)*Ntargets/2

        if Ntargets > 1:
            ## initialize velocity estimate vector
            v_hat = np.zeros((2,iter), dtype=float)

            k = 0
            for i in range(Ntargets-1):
                for j in range(i+1,Ntargets):

                    doppler = np.array([radar_doppler[i], radar_doppler[j]])
                    azimuth = np.array([radar_azimuth[i], radar_azimuth[j]])

                    v_hat[:,k] = self.doppler2BodyFrameVelocity(doppler, azimuth)

                    k+=1

            ## identify non-NaN solutions to uniquely-determined problem
            idx_nonNaN = np.isfinite(v_hat[1,:]);
            v_hat_nonNaN = v_hat[:,idx_nonNaN];

            ## does not math the size of the nonNaN vector in matlab!
            rospy.loginfo("v_hat_nonNaN.shape[1] = %d", v_hat_nonNaN.shape[1])
            rospy.loginfo("v_hat_nonNaN.size = %d", v_hat_nonNaN.size)

            if v_hat_nonNaN.shape[1] == 0:
                ## there exists no unique solution to the uniquely-determined
                ## problem for any two targets in the scan. This is the result of M
                ## being close to singular for all pairs of targets, i.e. the
                ## targets have identical angular locations.
                v_hat_all = float('nan')*np.ones((2,1))

        else:
            ## cannot solve uniquely-determined problem for a single target
            ## (solution requires 2 non-identical targets)
            v_hat_nonNaN = []
            v_hat_all = float('nan')*np.ones((2,1))

        rospy.loginfo("mean(v_hat_nonNaN,2) =" + str(np.mean(v_hat_nonNaN, axis=1)))

        if ( Ntargets > 2 ) and ( v_hat_nonNaN.shape[1] > 0 ):
            ## if there are more than 2 targets in the scan (resulting in a minimum
            ## of 3 estimates of the velocity vector), AND there exists at least one
            ## non-singular solution to the uniquely-determined problem

            ## remove k-sigma outliers from data and return sample mean as model
            sigma = np.std(v_hat_nonNaN, axis=1)    # sample std. dev.
            mu = np.mean(v_hat_nonNaN, axis=1)      # sample mean
            rospy.loginfo("mu =" + str(mu))

            rospy.loginfo("sigma =" + str(sigma))

            ## 2 targets will result in a single solution, and a variance of 0.
            ## k-sigma inliers should only be identified for more than 2 targets.
            k = 1;
            if sigma[0] > 0:
                idx_inlier_x = np.nonzero(np.absolute(v_hat_nonNaN[0,:]-mu[0]) < k*sigma[0]);
            else:
                idx_inlier_x = np.linspace(0,v_hat_nonNaN.shape[1]-1,v_hat_nonNaN.shape[1],dtype=int)

            if sigma[1] > 0:
                idx_inlier_y = np.nonzero(np.absolute(v_hat_nonNaN[1,:]-mu[1]) < k*sigma[1]);
            else:
                idx_inlier_y = np.linspace(0,v_hat_nonNaN.shape[1]-1,v_hat_nonNaN.shape[1],dtype=int)

            ## remove k-sigma outliers
            idx_inlier = reduce(np.intersect1d, (idx_inlier_x, idx_inlier_y))
            rospy.loginfo("idx_inlier = " + str(idx_inlier))

            model = np.mean(v_hat_nonNaN[:,idx_inlier], axis=1)
            v_hat_all = v_hat_nonNaN[:,idx_inlier]

        elif ( Ntargets > 1 ) and ( v_hat_nonNaN.shape[1] > 0 ):
            # there are 2 targets in the scan, AND their solution to the
            # uniquely-determined problem produced a non-singular matrix M
            model = v_hat
            v_hat_all = v_hat

        elif ( Ntargets > 1 ) and ( v_hat_nonNaN.shape[1] == 0 ):
            # there are 2 targets in the scan, AND their solution to the
            # uniquely-determined problem produced a singular matrix M, i.e. the
            # targets have identical angular locations.
            model = float('nan')*np.ones((2,1))
            v_hat_all = float('nan')*np.ones((2,1))

        else:
            # there is a single target in the scan, and the solution to the
            # uniquely-determined problem is not possible
            model = float('nan')*np.ones((2,1))

        # model = np.transpose([[0,0]])
        # model_all = np.zeros((2,iter), dtype=float)

        return model, v_hat_all
