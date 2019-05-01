#! /usr/bin/env python
"""
Author:         Carl Stahoviak
Date Created:   Apr 28, 2019
Last Edited:    Apr 28, 2019

Description:

"""

import rospy
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize
from radar_velocity_estimator.radar_doppler_model_2D import RadarDopplerModel2D

class OrthogonalDistanceRegression2D:

    def __init__(self):
        self.model = RadarDopplerModel2D()

        self.sigma_vr = 0.044
        self.sigma_theta = 0.0413
        self.d = self.sigma_vr/self.sigma_theta     # error variance ratio

        self.converge_thres = 0.0002    # ODR convergence threshold on step s
        self.maxIterations = 100        # max number of ODR iterations
        self.debug = False              # for comparison to MATLAB implementation


    def odr(self, radar_doppler, radar_azimuth, d, beta0, delta0, weights):
        """
        d - error variance ratio := sigma_vr / sigma_theta
        """
        Ntargets = delta0.shape[0]
        p = beta0.shape[0]

        S = 10*np.eye(p)        # s scaling matrix - 10 empirically chosen
        T = np.eye(Ntargets)    # t scaling matrix
        alpha = 0.001           # Lagrange multiplier

        ## initialize
        beta = beta0
        delta = delta0
        s = np.ones((p,), dtype=float)

        iter = 0
        while (np.abs(s[0]) > self.converge_thres) or (np.abs(s[1]) > self.converge_thres):
            # get Jacobian matrices
            G, V, D = self.getJacobian(radar_azimuth, delta, beta, weights, d)

            ## defined to simplify the notation in objectiveFunc
            P = np.matmul(V.conj().T,V) + np.matmul(D,D) + alpha*np.matmul(T,T)

            doppler_predicted = self.model.simulateRadarDoppler(beta, \
                    radar_azimuth, np.zeros((Ntargets,), dtype=float), delta)

            ## re-calculate epsilon
            eps = doppler_predicted - radar_doppler

            ## anonymous function defined within interation loop in order to use
            ## current values of G, V, D, eps and delta
            f = lambda param: self.objectiveFunc(param,G,V,D,P,eps,delta)

            soln = minimize(f,s)
            # print("soln =\n" + str(soln))
            s = soln.x
            t = np.matmul(-np.linalg.inv(P), np.matmul(V.conj().T,eps) + \
                    np.matmul(D,delta) + np.matmul(np.matmul(V.conj().T,G),s))

            # use s and t to iteratively update beta and delta, respectively
            beta = beta + np.matmul(S,s)
            delta = delta + np.matmul(T,t)

            # rospy.loginfo("[k, s] = " + str(np.array([[k, s[0], s[1]]]) ) )

            iter+=1
            if iter > self.maxIterations:
                break

        model = beta
        return model

    def objectiveFunc(self,s,G,V,D,P,eps,delta):
        if (G.shape[0] != V.shape[0]) and (G.shape[0] != V.shape[0]) \
            and (G.shape[0] != P.shape[0]) and (G.shape[1] != s.shape[0]):

            rospy.logerr('objectiveFunc: MATRIX SIZE MISMATCH')
        else:
            n = G.shape[0]

            # defined to simply the final form of the objective function, f
            f1 = fractional_matrix_power(np.eye(n) - \
                    np.matmul(np.matmul(V,np.linalg.inv(P)),V.conj().T), 0.5)
            f2 = fractional_matrix_power(np.eye(n) - \
                    np.matmul(np.matmul(V,np.linalg.inv(P)),V.conj().T), -0.5)
            f3 = np.matmul(V.conj().T,eps) + np.matmul(D,delta)
            f4 = -eps + np.matmul(np.matmul(V,np.linalg.inv(P)),f3)

            f = np.matmul(np.matmul(f1,G),s) - np.matmul(f2,f4)

            return np.linalg.norm(f)


    def getJacobian(self, X, delta, beta, weights, d):
        ## NOTE: We will use ODRPACK95 notation where the total Jacobian J has
        ## block components G, V and D:

        # J = [G,          V;
        #      zeros(n,p), D]

        # G - the Jacobian matrix of epsilon wrt/ beta and has no special properites
        # V - the Jacobian matrix of epsilon wrt/ delta and is a diagonal matrix
        # D - the Jacobian matrix of delta wrt/ delta and is a diagonal matrix

        # d - error variance ratio := sigma_epsilon/sigma_delta

        Ntargets = X.shape[0]   # X is a column vector of azimuth values
        p = beta.shape[0]

        # initialize
        G = np.zeros((Ntargets,p), dtype=float)
        V = np.zeros((Ntargets,Ntargets), dtype=float)
        D = np.zeros((Ntargets,Ntargets), dtype=float)

        for i in range(Ntargets):
            G[i,:] = np.array([np.cos(X[i] + delta[i]), np.sin(X[i] + delta[i])])
            V[i,i] = -beta[0]*np.sin(X[i] + delta[i]) + beta[1]*np.cos(X[i] + delta[i])

        G = np.matmul(np.diag(weights), G)
        V = np.matmul(np.diag(weights), V)
        D = d*np.diag(weights)

        return G, V, D

    def getWeights(self):
        pass

def test_odr():
    import pylab

    # init instance of ODR class
    odr = OrthogonalDistanceRegression2D()

    # number of simulated targets
    Ntargets = 10

    ## Generate Simulated Radar Measurements
    # hard-coded values
    velocity = np.array([[1.2], [0.75]])

    # simulated 'true' platform velocity
    min_vel = -2.5      # [m/s]
    max_vel = 2.5       # [m/s]
    # velocity = (max_vel-min_vel).*rand(2,1) + min_vel

    # create noisy simulated radar measurements
    radar_azimuth_bins = np.genfromtxt('1642_azimuth_bins.csv', delimiter=',')
    true_azimuth, true_doppler, radar_azimuth, radar_doppler = \
        odr.model.getSimulatedRadarMeasurements(Ntargets, velocity, \
            radar_azimuth_bins, odr.sigma_vr, debug=odr.debug)

    target_nums = np.linspace(0,Ntargets-1,Ntargets)
    target_nums = np.array(target_nums)[np.newaxis]
    radar_data = np.column_stack((target_nums.T, true_azimuth, radar_azimuth, \
        true_doppler, radar_doppler))
    # print('Radar Data:\n')
    # print("\t" + str(radar_data))

    ## Implement Estimation Schemes
    # get 'brute force' estimate of forward/lateral body-frame vel.
    model_bruteforce, _ = odr.model.getBruteForceEstimate(radar_doppler, radar_azimuth)
    print("Brute-Force Velocity Profile Estimation:")
    print("\t" + str(model_bruteforce))

    # get MLESAC (M-estimator RANSAC) model and inlier set
    # [ model_mlesac, inlier_idx ] = MLESAC( radar_doppler', ...
    #     radar_angle', sampleSize, maxDistance, conditionNum_thres );
    # fprintf('MLESAC Velocity Profile Estimation\n');
    # disp(model_mlesac)
    # fprintf('MLESAC Number of Inliers\n');
    # disp(sum(inlier_idx));

    # get Orthogonal Distance Regression (ODR) estimate - MLESAC seed

    # get Orthogonal Distance Regression (ODR) estimate - brute-force seed
    weights = (1/odr.sigma_vr)*np.ones((Ntargets,), dtype=float)
    if odr.debug:
        delta = np.ones((Ntargets,), dtype=float)*odr.sigma_theta
    else:
        delta = np.random.normal(0,odr.sigma_theta,(Ntargets,))
    model_odr = odr.odr( radar_doppler, radar_azimuth, odr.d, \
        model_bruteforce, delta, weights )
    print("ODR Velocity Profile Estimation - brute-force seed:");
    print("\t" + str(model_odr))

    RMSE_bruteforce     = np.sqrt(np.square(velocity - model_bruteforce))
    RMSE_odr_bruteforce = np.sqrt(np.square(velocity - model_odr))

    print("\nRMSE_brute_force = " + str(RMSE_bruteforce))
    print("RMSE_brute_force = " + str(RMSE_odr_bruteforce))


if __name__=='__main__':
    test_odr()
