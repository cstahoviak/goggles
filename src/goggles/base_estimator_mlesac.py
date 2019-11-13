"""
Author:         Carl Stahoviak
Date Created:   July 25, 2019
Last Edited:    July 25, 2019

Description:
Base Estimator class for MLESAC Regression.

"""
from __future__ import division
import numpy as np
from goggles.radar_utilities import RadarUtilities

class dopplerMLESAC():

    def __init__(self, model):
        ## ascribe doppler velocity model (2D, 3D) to the class
        self.model = model
        self.utils = RadarUtilities()

        ## define MLESAC parameters
        self.sample_size    = self.model.min_pts     # the minimum number of data values required to fit the model
        self.max_iterations = 40    # the maximum number of iterations allowed in the algorithm
        self.max_distance   = 0.15  # a threshold value for determining when a data point fits a model
        self.converge_thres = 10    # change in data log likelihood fcn required to indicate convergence

        self.param_vec_        = None   # body-frame velocity vector - to be estimated by MLESAC
        self.param_vec_mlesac_ = None   # array_like param_vec_ with shape (n,)
        self.param_vec_ols_    = None   # array_like param_vec_ with shape (n,)
        self.covariance_       = None   # covariance of parameter estimate, shape (p,p)

    ## model fit fcn
    def fit(self, data):
        self.param_vec_ = self.model.doppler2BodyFrameVelocity(data)
        return self

    ## distance(s) from data point(s) to model
    def distance(self, data):
        ## TODO: use residual function as part of computing the distances
        Ntargets = data.shape[0]
        p = self.sample_size

        radar_doppler   = data[:,0]      # [m/s]
        radar_azimuth   = data[:,1]      # [rad]
        radar_elevation = data[:,2]      # [rad]

        ## do NOT corrupt measurements with noise
        eps = np.zeros((Ntargets,), dtype=np.float32)
        delta = np.zeros(((p-1)*Ntargets,), dtype=np.float32)

        ## compute distances via residual
        eps_sq = np.square( self.residual(self.param_vec_, data) )
        distances = np.sqrt(eps_sq)

        ## distance per data point (column vector)
        return distances

    ## evaluate the data log likelihood of the data given the model - P(evidence | model)
    def score(self, data, type):
        ## TODO: use residual function as part of computing the score
        Ntargets = data.shape[0]
        p = self.sample_size

        radar_doppler  = data[:,0]
        radar_azimuth  = data[:,1]
        radar_elevaton = data[:,2]

        if type == 'mlesac':
            model = self.param_vec_mlesac_
        elif type == 'ols':
            model = self.param_vec_ols_
        else:
            model = self.param_vec_

        ## evaluate the data log-likelihood given the model
        eps_sq = np.square( self.residual(self.param_vec_, data) )
        score = -1/(2*self.model.sigma_vr**2)*np.sum(eps_sq)

        return score

    ## returns residuals vector (n,)
    def residual(self, X, data):
        Ntargets = data.shape[0]
        p = self.sample_size

        radar_doppler  = data[:,0]
        radar_azimuth  = data[:,1]
        radar_elevaton = data[:,2]

        doppler_predicted = self.model.simulateRadarDoppler(
            X, \
            np.column_stack((radar_azimuth,radar_elevaton)), \
            np.zeros((Ntargets,), dtype=np.float32), \
            np.zeros(((p-1)*Ntargets,), dtype=np.float32))

        eps = doppler_predicted - radar_doppler
        return eps

    ## returns Jacobain matrix, partial_eps/partial_beta (n,p)
    def jac(self, X, data):
        ## TODO: move this calculation over to the radar_doppler_model class
        Ntargets = data.shape[0]   # X is a column vector of azimuth values

        theta = data[:,1]       # azimuth angle column vector [rad]
        phi   = data[:,2]       # elevation angle column vector [rad]

        # initialize
        J = np.zeros((Ntargets,self.sample_size), dtype=np.float32)

        for i in range(Ntargets):
            if self.sample_size == 2:
                J[i,:] = np.array([np.cos(theta[i]),np.sin(theta[i])])

            elif self.sample_size == 3:
                J[i,:] = np.array([np.cos(theta[i])*np.cos(phi[i]), \
                                   np.sin(theta[i])*np.sin(phi[i]), \
                                   np.sin(phi[i]) ])
            else:
                raise ValueError("Jacobian Error")

        return J


    def is_data_valid(self, data):
        if data.shape[0] != self.sample_size:
            raise ValueError("data must be an 2x2 or 3x3 square matrix")

        if self.sample_size == 2:
            radar_doppler = data[:,0]
            radar_azimuth = data[:,1]

            numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)

            if numAzimuthBins > 1:
                is_valid = True
            else:
                is_valid = False

        elif self.sample_size == 3:
            radar_doppler   = data[:,0]
            radar_azimuth   = data[:,1]
            radar_elevation = data[:,2]

            numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)
            numElevBins = self.utils.getNumAzimuthBins(radar_elevation)

            if numAzimuthBins + numElevBins > 4:
                is_valid = True
            else:
                is_valid = False
        else:
            raise ValueError("data must be an Nx2 or Nx3 matrix")

        return is_valid
