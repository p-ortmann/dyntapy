import numpy as np
from dyntapy.dta.core.network_loading.link_models.i_ltm_cls import ILTMNetwork
from dyntapy.visualization import show_dynamic_network
from dyntapy.dta.core.supply import Network
from dyntapy.dta.core.time import SimulationTime
from numba import njit, prange, objmode
from numba.typed import List
from dyntapy.settings import dynamic_parameters
epsilon = dynamic_parameters.network_loading.epsilon


def travel_times_from_cumulatives(cvn_up: np.ndarray, cvn_down: np.ndarray, con_down: np.ndarray,
                                  free_flow_travel_times: np.ndarray, time_step: np.float, method='SF'):
    """

    Parameters
    ----------
    cvn_up : upstream cumulatives, dim tot_time_steps x links
    cvn_down : downstream cumulatives, dim tot_time_steps x links
    con_down : boolean, dim tot_time_steps x links
    method : str, ['SF','LI']

    Returns
    -------

    """
    U = cvn_up
    V = cvn_down
    y = np.zeros_like(U)
    travel_times = np.empty_like(U)
    for t in range(U.shape[0]):
        for link in range(U.shape[1]):
            if con_down[t, link]:
                y[t, link] = U[t, link] - U[t - 1, link]
                if t == 0:
                    y[t, link] = U[t, link]
                m_k = find_mk(U[t, link], V[:, link], t)
                travel_times[t, link] = 0


def find_mk(U_ak, V_a, k):
    # find last departure interval.
    for _k in range(k, V_a.size):
        if V_a[_k] >= U_ak:
            return _k
