#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   July 23, 2019
Last Edited:    July 24, 2019

Description:

"""
from __future__ import division
import time
import numpy as np
from functools import partial, reduce

from scipy.optimize import least_squares
from goggles.radar_utilities import RadarUtilities
from goggles.radar_doppler_model_2D import RadarDopplerModel2D
from goggles.radar_doppler_model_3D import RadarDopplerModel3D
from goggles.base_estimator_mlesac import dopplerMLESAC
# from sklearn.linear_model import RANSACRegressor
# from goggles.base_estimator import dopplerRANSAC

class MLESAC:

    def __init__(self, base_estimator, report_scores=False, ols_flag=False, get_covar=False):
        self.estimator_ = base_estimator

        self.inliers_  = None     # inlier data points
        self.scores_   = None     # data log likelihood associated with each iteration
        self.iter_     = None     # number of iterations until convergence

        self.report_scores = report_scores  # report data log likelihood of each iteration
        self.ols_flag      = ols_flag       # enable OLS solution on inlier set
        self.get_covar     = get_covar      # return estimate covariance?

    def mlesac(self, data):

        Ntargets = data.shape[0]    # data.shape = (Ntargets,p)

        bestScore = -np.inf
        bestInliers = []
        bestModel = []
        scores = []

        dll_incr = np.inf           # increase in data log likelihood function
        iter = 0                    # algorithm iteration Number

        while np.abs(dll_incr) > self.estimator_.converge_thres and \
            iter < self.estimator_.max_iterations:

            ## randomly sample from data
            idx = np.random.randint(Ntargets,high=None,size=(self.estimator_.sample_size,))
            sample = data[idx,:]

            is_valid = self.estimator_.is_data_valid(sample)
            if is_valid:
                ## estimate model parameters from sampled data points
                param_vec_temp = self.estimator_.param_vec_
                self.estimator_.fit(sample)

                ## score the model - evaluate the data log likelihood fcn
                score = self.estimator_.score(data,type=None)

                if score > bestScore:
                    ## this model better explains the data
                    distances = self.estimator_.distance(data)

                    dll_incr    = score - bestScore     # increase in data log likelihood fcn
                    bestScore   = score
                    bestInliers = np.nonzero((distances < self.estimator_.max_distance))

                    if self.report_scores:
                        scores.append(score)

                    # evaluate stopping criteria - not yet used
                    # Ninliers = sum(bestInliers)
                    # w = Ninliers/Ntargets
                    # k = np.log(1-0.95)*np.log(1-w**2)
                else:
                    ## candidate param_vec_ did NOT have a higher score
                    self.estimator_.param_vec_ = param_vec_temp

                iter+=1
                # print("iter = " + str(iter) + "\tscore = " + str(score))
            else:
                ## do nothing - cannot derive a valid model from targets in
                ## the same azimuth/elevation bins

                # print("mlesac: INVALID DATA SAMPLE")
                pass

        self.estimator_.param_vec_mlesac_ = self.estimator_.param_vec_
        self.inliers_ = reduce(np.intersect1d,(bestInliers))
        self.scores_ = np.array(scores)
        self.iter_ = iter

        ## get OLS solution on inlier set
        if self.ols_flag:
            # callable = partial(self.estimator_.residual, data=data)
            # ols_soln = least_squares(callable, self.estimator_.param_vec_)

            ols_soln = least_squares(self.estimator_.residual, \
                self.estimator_.param_vec_, self.estimator_.jac, \
                kwargs={"data": data[self.inliers_,:]})
            self.estimator_.param_vec_ols_ = ols_soln.x

            ## score both estimates
            score_mlesac = self.estimator_.score(data[self.inliers_,:],'mlesac')
            score_ols = self.estimator_.score(data[self.inliers_,:],'ols')

            if score_ols > score_mlesac:
                ## OLS solution is better than MLESAC solution
                self.estimator_.param_vec_ = self.estimator_.param_vec_ols_

                ##TODO: maybe re-evaulate inliers??
            else:
                ## do nothing - MLESAC solution is better than OLS solution
                pass

            if self.get_covar:
                eps = ols_soln.fun      # residual vector at solution
                jac = ols_soln.jac      # modified Jacobian matrix at solution

                self.estimator_.covariance_ = np.matmul(eps.T,eps) * \
                    np.linalg.inv(np.matmul(jac.T,jac))

        else:
            self.estimator_.param_vec_ols_ = \
                float('nan')*np.ones((self.estimator_.sample_size,))

        return


def test(model):
    ## define MLESAC parameters
    report_scores = False
    ols_flag = True
    get_covar = True

    # init instance of MLESAC class
    base_estimator_mlesac = dopplerMLESAC(model)
    mlesac = MLESAC(base_estimator_mlesac, report_scores, ols_flag, get_covar)

    ## instantiate scikit-learn RANSAC object with base_estimator class object
    # base_estimator_ransac = dopplerRANSAC(model=model)
    # ransac = RANSACRegressor(base_estimator=base_estimator_ransac, \
    #     min_samples=base_estimator_ransac.sample_size, \
    #     residual_threshold=base_estimator_ransac.max_distance, \
    #     is_data_valid=base_estimator_ransac.is_data_valid, \
    #     max_trials=base_estimator_ransac.max_iterations, \
    #     loss=base_estimator_ransac.loss)

    ## outlier std deviation
    sigma_vr_outlier = 1.5

    radar_angle_bins = np.genfromtxt('../../data/1642_azimuth_bins.csv', delimiter=',')

    ## simulated 'true' platform velocity range
    min_vel = -2.5      # [m/s]
    max_vel = 2.5       # [m/s]

    ## number of simulated targets
    Ninliers = 125
    Noutliers = 35

    ## generate truth velocity vector
    velocity = (max_vel-min_vel)*np.random.random((base_estimator_mlesac.sample_size,)) + min_vel

    ## create noisy INLIER  simulated radar measurements
    _, inlier_data = model.getSimulatedRadarMeasurements(Ninliers, \
        velocity,radar_angle_bins,model.sigma_vr)

    ## create noisy OUTLIER simulated radar measurements
    _, outlier_data = model.getSimulatedRadarMeasurements(Noutliers, \
        velocity,radar_angle_bins,sigma_vr_outlier)

    ## combine inlier and outlier data sets
    Ntargets = Ninliers + Noutliers
    radar_doppler = np.concatenate((inlier_data[:,0],outlier_data[:,0]),axis=0)
    radar_azimuth = np.concatenate((inlier_data[:,1],outlier_data[:,1]),axis=0)
    radar_elevation = np.concatenate((inlier_data[:,2],outlier_data[:,2]),axis=0)

    ## get MLESAC estimate + inlier set
    start_time = time.time()
    radar_data = np.column_stack((radar_doppler,radar_azimuth,radar_elevation))
    mlesac.mlesac(radar_data)
    model_mlesac = mlesac.estimator_.param_vec_mlesac_
    model_ols = mlesac.estimator_.param_vec_ols_
    inliers = mlesac.inliers_
    time_mlesac = time.time() - start_time

    ## get scikit-learn RANSAC estimate + inlier set
    ## NOTE: DOES NOT WORK YET
    # start_time = time.time()
    # ransac.fit(radar_data)
    # model_ransac = np.squeeze(self.ransac.estimator_.param_vec_)
    # inlier_mask = self.ransac.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)
    # time_ransac = time.time() - start_time

    print("\nMLESAC Velocity Profile Estimation:\n")
    print("Truth\t MLESAC\t\tMLESAC+OLS")
    for i in range(base_estimator_mlesac.sample_size):
        print(str.format('{0:.4f}',velocity[i]) + "\t " + str.format('{0:.4f}',model_mlesac[i]) \
              + " \t" + str.format('{0:.4f}',model_ols[i]))

    rmse_mlesac = np.sqrt(np.mean(np.square(velocity - model_mlesac)))
    print("\nRMSE (MLESAC)\t= " + str.format('{0:.4f}',rmse_mlesac) + " m/s")

    if mlesac.ols_flag:
        rmse_ols = np.sqrt(np.mean(np.square(velocity - model_ols)))
        print("RMSE (OLS)\t= " + str.format('{0:.4f}',rmse_ols) + " m/s")

    print("\nExecution Time = %s" % time_mlesac)

def test_montecarlo(model):
    pass

if __name__=='__main__':
    # model = RadarDopplerModel2D()
    model = RadarDopplerModel3D()
    test(model)
