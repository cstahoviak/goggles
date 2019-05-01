"""
Author:         Carl Stahoviak
Date Created:   Apr 30, 2019
Last Edited:    Apr 30, 2019

Description:
Base Estimator class for RANSAC Regression.

"""

import rospy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from radar_velocity_estimator.radar_utilities import RadarUtilities

class dopplerRANSAC(BaseEstimator, RegressorMixin):

    def __init__(self, model):
        # ascribe doppler velocity model (2D, 3D) to the class
        self.model = model
        self.utils = RadarUtilities()

        # body-frame velocity vector - to be estimated by RANSAC
        self.param_vec_ = None

    # fit(X,y): Fit model to given training data and target values
    def fit(self, X, y):
        radar_azimuth = np.squeeze(X)
        radar_doppler = np.squeeze(y)

        model = self.model.doppler2BodyFrameVelocity(radar_doppler, radar_azimuth)
        # rospy.loginfo("fit: model = " + str(model))
        self.param_vec_ = model
        return self

    # predict(X): Returns predicted values used to compute residual error using loss function
    def predict(self, X):
        radar_azimuth = np.squeeze(X)
        Ntargets = radar_azimuth.shape[0]

        doppler_predicted = self.model.simulateRadarDoppler(self.param_vec_, \
                                radar_azimuth, np.zeros((Ntargets,), dtype=float), \
                                np.zeros((Ntargets,), dtype=float))

        # rospy.loginfo("predict: doppler_predicted = \n" + str(doppler_predicted))
        return doppler_predicted

    def loss(self, y, y_pred):
        dist = np.sqrt(np.square(np.squeeze(y) - y_pred))
        # rospy.loginfo("loss: dist.shape = " + str(dist.shape))
        return dist


    # Don't need to define score(X,y) if inherited from RegressorMixin... I think
    # score(X,y): Returns the mean accuracy on the given test data, which is used
    # for the stop criterion defined by stop_score
    # def score(self, X, y):
    #     radar_azimuth = np.squeeze(X)
    #     radar_doppler = np.squeeze(y)
    #     Ntargets = radar_azimuth.shape[0]
    #
    #     rospy.loginfo("score: radar_azimuth.shape = " + str(radar_azimuth.shape))
    #     rospy.loginfo("score: radar_doppler.shape = " + str(radar_doppler.shape))
    #
    #     doppler_predicted = self.model.simulateRadarDoppler(self.param_vec_, \
    #                             radar_azimuth, np.zeros((Ntargets,), dtype=float), \
    #                             np.zeros((Ntargets,), dtype=float))
    #
    #     rospy.loginfo("score: radar_doppler = " + str(radar_doppler))
    #     rospy.loginfo("score: doppler_predicted = " + str(doppler_predicted))
    #
    #     dist = np.sqrt(np.square(radar_doppler - doppler_predicted))
    #     rospy.loginfo("score: dist = \n" + str(dist))
    #     return np.mean(dist, axis=0)


    def is_data_valid(self, X, y):
        radar_azimuth = np.squeeze(X)
        radar_doppler = np.squeeze(y)

        numAzimuthBins = self.utils.getNumAzimuthBins(radar_azimuth)
        # rospy.loginfo("is_data_valid: numAzimuthBins = " + str(numAzimuthBins))

        if numAzimuthBins > 1:
            is_valid = True
        else:
            is_valid = False

        return is_valid
