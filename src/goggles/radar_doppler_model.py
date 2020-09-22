from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from goggles.radar_utilities import RadarUtilities

nparr = np.ndarray


class RadarDopplerModel(ABC):
    """
    A RadarDopplerModel provides both forward and inverse measurement models
    for the generation simulated radar data, and the solution of the uniquely-
    determined problem (solving for the model parameters given a set of min_pts
    measurements), respectively.

    Additionally, it provides tools (getSimulatedRadarMeasurements,
    generatePointcloud) for the generation of simulated radar measurements.
    """
    def __init__(self):
        # radar utility functions
        self._utils = RadarUtilities()

        # define radar uncertainty parameters
        self._sigma_vr = 0.444          # [m/s]
        self._sigma_theta = 0.0426      # [rad]
        self._sigma_phi = None          # [rad]
        self._sigma = None

        # define ODR error variance ratio
        self._d = None

        # minimum number of points (measurements) to fit the model
        self.min_pts = None

        # define the default field-of-view (FOV) of the sensor
        # TODO: lookup the default FOV for the AWR1843 and set values here
        self._fov_az = None
        self._fov_elev = None

    @abstractmethod
    def doppler2BodyFrameVelocity(self, data: nparr) -> nparr:
        """
        The inverse measurement model: measurements -> model.
        Estimates the model parameters (the components of the ego-velocity
        vector) by solving the uniquely-determined problem for a set of
        measurements of size min_pts.

        Args:
            data: radar measurements (min_pts, min_pts)

        Returns:
            model: estimated ego-velocity vector (min_pts, )
        """
        pass

    @abstractmethod
    def simulateRadarDoppler(self, model: nparr, data: nparr, eps: nparr,
                             delta: nparr) -> nparr:
        """
        The generative (forward) measurement model: model -> measurements
        Generates a set of noisy radar measurements [azimuth/elevation, Doppler
        velocity] given an estimate of the model parameters (the components of
        the ego-velocity vector).

        Args:
            model: ego-velocity vector (min_pts, )
            data: radar measurements (Ntargets, min_pts)
            eps: Doppler velocity additive noise vector (Ntargets, )
            delta: azimuth/elevation additive noise vector (Ntargets, min_pts-1)

        Returns:
            radar_doppler: predicted Doppler velocity measurements (Ntargets, )
        """
        pass

    @abstractmethod
    def getSimulatedRadarMeasurements(self, Ntargets: int, model: nparr,
                                      radar_angle_bins: nparr, sigma_vr: float,
                                      debug: bool = False) -> \
            Tuple[nparr, nparr]:
        """
        Generates a set simulated radar measurements (Ntargets, min_pts) and
        returns a set of truth data and a set of noisy data.

        Args:
            Ntargets: number of simulated targets
            model: ego-velocity vector (min_pts, )
            radar_angle_bins: an (Nbins, ) vector of azimuth locations in which
            angle-of-arrival FFT processing bins a target's azimuth value
            sigma_vr: Doppler velocity uncertainty
            debug: use constant value for additive Doppler noise?

        Returns:
            data_truth: the truth data, shape (Ntargets, ?)
            data_sim: the simulated (noisy) radar data (Ntargets, ?)
        """
        pass

    @abstractmethod
    def generatePointcloud(self, Ntargets: int, fov_az: float = None,
                           fov_elev: float = None) -> nparr:
        """
        Generates a randomly distributed set of radar targets within the FOV of
        the sensor.
        TODO: update this function to only generate targets within a specified
            azimuth and elevation FOV - these ranges taken as inputs.

        Args:
            Ntargets: number of simulated targets
            fov_az: azimuth field of view is +/- fov_az
            fov_elev: elevation field of view is +/- fov_elev

        Returns:
            pointcloud: the simulated pountcloud (Ntargets, min_pts)
        """
        pass
