"""
Author:         Carl Stahoviak
Date Created:   Apr 22, 2019
Last Edited:    Apr 22, 2019

Description:

"""

from __future__ import division
import rospy
import numpy as np
from goggles.radar_utilities import RadarUtilities

class RadarDopplerModel2D:

    def __init__(self):
        self.utils = RadarUtilities()

        ## define radar uncertainty parameters
        self.sigma_vr = 0.044               # [m/s]
        self.sigma_theta = 0.0426           # [rad]
        self.sigma = self.sigma_theta

        ## define ODR error variance ratio
        self.d = self.sigma_vr / self.sigma_theta

        self.min_pts = 2    # minimum number of data points to fit the model of close data values required to assert that a model fits well to data

    # defined for RANSAC - not used
    def fit(self, data):
        radar_doppler = data[:,0]   # [m/s]
        radar_azimuth = data[:,1]   # [rad]

        model = self.doppler2BodyFrameVelocity(radar_doppler, radar_azimuth)
        return model


    # defined for RANSAC - not used
    def get_error(self, data, model):
        ## number of targets in scan
        Ntargets = data.shape[0]
        error = np.zeros((Ntargets,), dtype=np.float32)

        radar_doppler = data[:,0]   # [m/s]
        radar_azimuth = data[:,1]   # [rad]

        ## do NOT corrupt measurements with noise
        eps = np.zeros((Ntargets,), dtype=np.float32)
        delta = np.zeros((Ntargets,), dtype=np.float32)

        ## radar doppler generative model
        doppler_predicted = self.simulateRadarDoppler(model, radar_azimuth, eps, delta)

        for i in range(Ntargets):
            error[i] = np.sqrt((doppler_predicted[i] - radar_doppler[i])**2)

        ## error per data point (column vector)
        return error
        # return np.squeeze(error)


    # inverse measurement model: measurements->model
    def doppler2BodyFrameVelocity(self, data):
        radar_doppler = data[:,0]       # doppler velocity [m/s]
        radar_azimuth = data[:,1]       # azimuth angle column vector [rad]

        numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)

        # rospy.loginfo("doppler2BodyFrameVelocity: numAzimuthBins = %d", numAzimuthBins)
        # rospy.loginfo(['{0:5.4f}'.format(i) for i in radar_azimuth])    # 'list comprehension'
        # # rospy.loginfo(['{0:5.4f}'.format(i) for i in radar_doppler])    # 'list comprehension'

        if numAzimuthBins > 1:
           ## solve uniquely-determined problem for pair of targets (i,j)
            M = np.array([[np.cos(radar_azimuth[0]), np.sin(radar_azimuth[0])], \
                          [np.cos(radar_azimuth[1]), np.sin(radar_azimuth[1])]])

            b = np.array([[radar_doppler[0]], \
                          [radar_doppler[1]]])

            model = np.squeeze(np.linalg.solve(M,b))
        else:
            model = float('nan')*np.ones((2,))

        return model


    # measurement generative (forward) model: model->measurements
    def simulateRadarDoppler(self, model, data, eps, delta):
        Ntargets = data.shape[0]
        radar_doppler = np.zeros((Ntargets,), dtype=np.float32)

        radar_azimuth = data[:,0]

        for i in range(Ntargets):
            ## add measurement noise distributed as N(0,sigma_theta_i)
            theta = radar_azimuth[i] + delta[i]

            ## add meaurement noise epsilon distributed as N(0,sigma_vr)
            radar_doppler[i] = model[0]*np.cos(theta) + model[1]*np.sin(theta) + eps[i]

        return radar_doppler


    def getBruteForceEstimate(self, radar_doppler, radar_azimuth):
        ## number of targets in scan
        Ntargets = radar_doppler.shape[0]
        iter = (Ntargets-1)*Ntargets/2

        if Ntargets > 1:
            ## initialize velocity estimate vector
            v_hat = np.zeros((2,iter), dtype=np.float32)

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
                v_hat_all = float('nan')*np.ones((2,))

        else:
            ## cannot solve uniquely-determined problem for a single target
            ## (solution requires 2 non-identical targets)
            v_hat_nonNaN = []
            v_hat_all = float('nan')*np.ones((2,))

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
            model = float('nan')*np.ones((2,))
            v_hat_all = float('nan')*np.ones((2,))

        else:
            # there is a single target in the scan, and the solution to the
            # uniquely-determined problem is not possible
            model = float('nan')*np.ones((2,))

        return model, v_hat_all

    def getSimulatedRadarMeasurements(self, Ntargets, model, radar_azimuth_bins, \
                                        sigma_vr, debug=False):

        radar_azimuth = np.zeros((Ntargets,), dtype=np.float32)

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

        true_elevation = np.zeros((Ntargets,))
        radar_elevation = np.zeros((Ntargets,))

        ## define AGWN vector for doppler velocity measurements
        if debug:
            eps = np.ones((Ntargets,), dtype=np.float32)*sigma_vr
        else:
            eps = np.random.normal(0,sigma_vr,(Ntargets,))

        ## get true radar doppler measurements
        true_doppler = self.simulateRadarDoppler(model, np.column_stack((true_azimuth,true_elevation)), \
            np.zeros((Ntargets,), dtype=np.float32), np.zeros((Ntargets,), dtype=np.float32))

        # get noisy radar doppler measurements
        radar_doppler =  self.simulateRadarDoppler(model, np.column_stack((radar_azimuth,radar_elevation)), \
            eps, np.zeros((Ntargets,), dtype=np.float32))

        data_truth = np.column_stack((true_doppler,true_azimuth,true_elevation))
        data_sim = np.column_stack((radar_doppler,radar_azimuth,radar_elevation))

        return data_truth, data_sim
