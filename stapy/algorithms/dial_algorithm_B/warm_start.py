#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
from stapy.assignment import StaticAssignment


class DialBResults:
    def __init__(self, demand_dict, flows, bush_flows, topological_orders, adjacency, costs, derivatives):
        self.demand_dict=demand_dict
        self.flows = flows
        self.bush_flows = bush_flows
        self.topological_orders = topological_orders
        self.adjacency = adjacency
        self.costs = costs
        self.derivatives = derivatives

    def get_state(self):
        return self.flows, self.bush_flows, self.topological_orders, self.adjacency, self.derivatives, self.costs

    def update_bushes(self, ff_tts=None, altered_demand_dict=None):
        assert ff_tts or altered_demand_dict is not None
        pass
        # for i in self.demand_dict:
        #     costs = __bpr_cost(capacities=link_capacities, ff_tts=link_ff_times, flows=flows)
        #     destinations = demand_dict[i][0]
        #     demands = demand_dict[i][1]
        #     self.
