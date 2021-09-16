from numba import njit
from dyntapy.dta.core.time import SimulationTime
import numpy as np


@njit(cache=True)
# potentially using numbas optional ..
def treiber_helbing_asm(measurements, X, sampling_rate, delta_t, delta_x, speed_measurements=None):
    """
    as shown in Treiber, Martin, and Dirk Helbing. "An adaptive smoothing method
    for traffic state identification from incomplete information."
     Interface and Transport Dynamics.Springer, Berlin, Heidelberg, 2003. 343-360.

    Parameters
    ----------
    speed_measurements
    X: array, 1D detector locations along the x-axis, assumed to be monotonously increasing
    measurements: dim detector_locations x tot_aggregation_intervals
    sampling_rate: temporal aggregations of the detector data in minutes
    delta_x: float, requested spatial output granularity in km
    delta_t: float, requested temporal output granularity in minutes


    Returns
    -------
    array with estimated values at desired spatial and temporal granularity
    """
    if not np.all(X[1:] >= X[:-1], axis=1):
        raise ValueError('detector locations are assumed to be monotonously increasing')

    if speed_measurements is None:
        speed_measurements = measurements
    # detector data can be average velocity, vehicle flow, occupancy or traffic density and describe these quantities
    # either for a single lane or be an average across different lanes
    tot_detectors = measurements.shape()[0]
    tot_aggregation_intervals = measurements.shape()[1]
    # parameters for smoothing
    sigma = 0.6  # km, range of spatial smoothing in x
    tau = 1.1  # minutes, range of temporal smoothing in t
    c_free = 80  # km/h, propagation velocity of perturbations in free traffic
    c_cong = -15  # km/h, propagation velocity of perturbations in congested traffic
    v_c = 60  # km/h, crossover from free to congested traffic
    delta_v = 20  # km/h, width of the transition region

    def cong_kernel(x, t):
        np.exp(-np.abs(x) / sigma - np.abs(t - x / c_cong) / tau)

    def free_kernel(x, t):
        np.exp(-np.abs(x) / sigma - np.abs(t - x / c_free) / tau)

    def normalization_cong(detector_locations, tot_detectors, tot_aggregation_intervals, x, t):
        # left here for better readability, integrated in full z_cong calculation below
        factor = 0
        for detector in range(tot_detectors):
            for interval in range(tot_aggregation_intervals):
                factor += cong_kernel(detector_locations[detector] - x, interval * sampling_rate - t)
        return factor

    def normalization_free(detector_locations, tot_detectors, tot_aggregation_intervals, x, t):
        # left here for better readability, integrated in full z_free calculation below
        factor = 0
        for detector in range(tot_detectors):
            for interval in range(tot_aggregation_intervals):
                factor += free_kernel(detector_locations[detector] - x, interval * sampling_rate - t)
        return factor

    def z_cong(x, t, data, detector_locations, tot_detectors, tot_aggregation_intervals):
        factor = 0
        for detector in range(tot_detectors):
            for interval in range(tot_aggregation_intervals):
                factor += cong_kernel(detector_locations[detector] - x, interval * sampling_rate - t)
        _z_cong = 0
        for detector in range(tot_detectors):
            for interval in range(tot_aggregation_intervals):
                _z_cong += cong_kernel(detector_locations[detector] - x, interval * sampling_rate - t) * data[
                    detector, interval]
        return _z_cong / factor

    def z_free(x, t, data, detector_locations, tot_detectors, tot_aggregation_intervals):
        factor = 0
        for detector in range(tot_detectors):
            for interval in range(tot_aggregation_intervals):
                factor += free_kernel(detector_locations[detector] - x, interval * sampling_rate - t)
        _z_free = 0
        for detector in range(tot_detectors):
            for interval in range(tot_aggregation_intervals):
                _z_free += free_kernel(detector_locations[detector] - x, interval * sampling_rate - t) * data[
                    detector, interval]
        return _z_free / factor

    def w(x, t, v_star):
        # v_star is the specific value for x,t given externally
        return 0.5 * (1 + np.tanh(
            v_c - v_star / delta_v)
                      )

    out_time_dim = np.arange(0, (sampling_rate * tot_aggregation_intervals), delta_t)
    out_space_dim = np.arange(X[0], X[-1], delta_x)
