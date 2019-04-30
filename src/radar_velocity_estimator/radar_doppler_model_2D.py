"""
Author:         Carl Stahoviak
Date Created:   Apr 22, 2019
Last Edited:    Apr 22, 2019

Description:

"""

import rospy
import numpy as np
from radar_velocity_estimator.radar_utilities import RadarUtilities

class RadarDopplerModel2D:

    def __init__(self):
        self.utils = RadarUtilities()

        ## define RANSAC parameters
        self.sampleSize    = 2      # the minimum number of data values required to fit the model
        self.maxIterations = 1000   # the maximum number of iterations allowed in the algorithm
        self.maxDistance   = 0.1    # a threshold value for determining when a data point fits a model
        self.minPts        = 2      # the number of close data values required to assert that a model fits well to data

    # defined for RANSAC
    def fit(self, data):
        radar_doppler = data[:,0]   # [m/s]
        radar_azimuth = data[:,1]   # [rad]

        model = self.doppler2BodyFrameVelocity(radar_doppler, radar_azimuth)
        # rospy.loginfo("(ransac_fit) type(model) = " + str(type(model)))
        # rospy.loginfo("(ransac_fit) model.shape = " + str(model.shape))
        # rospy.loginfo("(ransac_fit) model = " + str(model))
        return model


    # defined for RANSAC
    def get_error(self, data, model):
        ## number of targets in scan
        Ntargets = data.shape[0]
        error = np.zeros((Ntargets,1), dtype=float)

        radar_doppler = data[:,0]   # [m/s]
        radar_azimuth = data[:,1]   # [rad]

        ## do NOT corrupt measurements with noise
        eps = np.zeros((Ntargets,1), dtype=float)
        delta = np.zeros((Ntargets,1), dtype=float)

        ## radar doppler generative model
        doppler_predicted = self.simulateRadarDoppler(model, radar_azimuth, eps, delta)

        for i in range(Ntargets):
            error[i] = np.sqrt((doppler_predicted[i] - radar_doppler[i])**2)

        ## error per data point (column vector)
        # rospy.loginfo("(get_error) error.shape = " + str(error.shape))
        # rospy.loginfo("error = " + str(error))
        return np.squeeze(error)


    # inverse measurement model: measurements->model
    def doppler2BodyFrameVelocity(self, radar_doppler, radar_azimuth):
        numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)

        if numAzimuthBins > 1:
           ## solve uniquely-determined problem for pair of targets (i,j)
            M = np.array([[np.cos(radar_azimuth[0]), np.sin(radar_azimuth[0])], \
                          [np.cos(radar_azimuth[1]), np.sin(radar_azimuth[1])]])

            b = np.array([[radar_doppler[0]], \
                          [radar_doppler[1]]])

            model = np.linalg.solve(M,b)
        else:
            model = float('nan')*np.ones((2,1))

        return model


    # measurement generative (forward) model: model->measurements
    def simulateRadarDoppler(self, model, radar_azimuth, eps, delta):
        # print("simulateRadarDoppler: radar_azimuth.shape = " + str(radar_azimuth.shape))
        Ntargets = radar_azimuth.shape[0]
        radar_doppler = np.zeros((Ntargets,), dtype=float)

        for i in range(Ntargets):
            ## add measurement noise distributed as N(0,sigma_theta_i)
            theta = radar_azimuth[i] + delta[i]

            radar_doppler[i] = model[0]*np.cos(theta) + model[1]*np.sin(theta)

            ## add meaurement noise distributed as N(0,sigma_vr)
            radar_doppler[i] = radar_doppler[i] + eps[i]

        return radar_doppler


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

                    model = self.doppler2BodyFrameVelocity(doppler, azimuth)
                    ## don't understand why I need to transpose an ndarray of shape (2,1)
                    v_hat[:,k] = np.transpose(model)

                    k+=1

            ## identify non-NaN solutions to uniquely-determined problem
            idx_nonNaN = np.isfinite(v_hat[1,:])
            v_hat_nonNaN = v_hat[:,idx_nonNaN]

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

        if ( Ntargets > 2 ) and ( v_hat_nonNaN.shape[1] > 0 ):
            ## if there are more than 2 targets in the scan (resulting in a minimum
            ## of 3 estimates of the velocity vector), AND there exists at least one
            ## non-singular solution to the uniquely-determined problem

            ## remove k-sigma outliers from data and return sample mean as model
            sigma = np.std(v_hat_nonNaN, axis=1)    # sample std. dev.
            mu = np.mean(v_hat_nonNaN, axis=1)      # sample mean

            ## 2 targets will result in a single solution, and a variance of 0.
            ## k-sigma inliers should only be identified for more than 2 targets.
            k = 1
            if sigma[0] > 0:
                idx_inlier_x = np.nonzero(np.abs(v_hat_nonNaN[0,:]-mu[0]) < k*sigma[0])
            else:
                idx_inlier_x = np.linspace(0,v_hat_nonNaN.shape[1]-1,v_hat_nonNaN.shape[1],dtype=int)

            if sigma[1] > 0:
                idx_inlier_y = np.nonzero(np.abs(v_hat_nonNaN[1,:]-mu[1]) < k*sigma[1])
            else:
                idx_inlier_y = np.linspace(0,v_hat_nonNaN.shape[1]-1,v_hat_nonNaN.shape[1],dtype=int)

            ## remove k-sigma outliers
            idx_inlier = reduce(np.intersect1d, (idx_inlier_x, idx_inlier_y))
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

        return model, v_hat_all

    def getSimulatedRadarMeasurements(self, Ntargets, model, radar_azimuth_bins, \
                                      sigma_vr, debug=False):
        radar_azimuth = np.zeros((Ntargets,), dtype=float)

        # simulated true target angles
        min_azimuth = np.deg2rad(-75)   # [rad]
        max_azimuth = np.deg2rad(75)    # [rad]
        if debug:
            true_azimuth = np.linspace(min_azimuth, max_azimuth, Ntargets)
        else:
            true_azimuth = (max_azimuth-min_azimuth)*np.random.random(Ntargets,) + min_azimuth

        # bin angular data
        for i in range(Ntargets):
            bin_idx = (np.abs(radar_azimuth_bins - true_azimuth[i])).argmin()
            ## could Additionally add some noise here
            radar_azimuth[i] = radar_azimuth_bins[bin_idx]

        ## define AGWN vector for doppler velocity measurements
        if debug:
            eps = np.ones((Ntargets,), dtype=float)*sigma_vr
        else:
            eps = np.random.normal(0,sigma_vr,(Ntargets,))

        ## get true radar doppler measurements
        true_doppler = self.simulateRadarDoppler(model, true_azimuth, \
            np.zeros((Ntargets,), dtype=float), np.zeros((Ntargets,), dtype=float))

        # get noisy radar doppler measurements
        radar_doppler =  self.simulateRadarDoppler(model, radar_azimuth, \
            eps, np.zeros((Ntargets,), dtype=float))

        return true_azimuth, true_doppler, radar_azimuth, radar_doppler
