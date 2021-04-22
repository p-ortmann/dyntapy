#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
from dtapy.core.time import SimulationTime
import numpy as np


def get_duet_result(flows: np.ndarray, costs: np.ndarray,
                    start_time: int, simulation_time: SimulationTime):
    """

    Parameters
    ----------
    flows : tot_time_steps x tot_links
    costs : tot_time_steps x tot_links
    start_time : int, start of simulation in time since epoch (seconds)
    simulation_time : SimulationTime

    Returns
    -------

    """

    simulation_time_intervals = (np.arange(simulation_time.start, simulation_time.end,
                                           simulation_time.step_size)*3600+start_time).tolist()
    duet_result = {
        "metadata": {
            "type": "links",
            "status": "ok",
            "error_msg": "",
            "simulation_time_intervals": simulation_time_intervals,
            "road_network_geom_url": f"",
            "units": {
                "cost": "hour",
                "flow": "vehicles per time_interval ",
            }
        }
    }

    result = []
    for link, (flow, cost) in enumerate(zip(flows, costs)):
        result.append({
            "id": link,
            "flow": flows[:, link].tolist(),
            "cost": cost[:, link].tolist()
        })

    duet_result["result"] = result
    return [duet_result]
