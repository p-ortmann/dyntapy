User reference
==============

dyntapy.assignments
----------------------------------------

.. automodule:: dyntapy.assignments
    :members:

dyntapy.demand_data
----------------------------------------

.. automodule:: dyntapy.demand_data
    :members:

dyntapy.supply_data
----------------------------------------

.. automodule:: dyntapy.supply_data
    :members:

dyntapy.demand
----------------------------------------

.. automodule:: dyntapy.demand
    :members:
   
dyntapy.supply
----------------------------------------

.. automodule:: dyntapy.supply
    :members:

dyntapy.graph_utils
----------------------------------------

.. automodule:: dyntapy.graph_utils
    :members:

dyntapy.results
----------------------------------------

.. automodule:: dyntapy.results
    :members:

dyntapy.csr
----------------------------------------

.. automodule:: dyntapy.csr
    :members:

dyntapy.visualization
----------------------------------------

.. automodule:: dyntapy.visualization
    :members:

Debugging Assignments
----------------------------------------

By default most of dyntapy's assignment algorithms utilize `Numba`_ to accelerate
computations. It is not possible to use breakpoints
inside of code that bas been JIT-compiled. We first need to disable JIT compilation
to do so:

.. _Numba: https://numba.pydata.org/

>>> from numba import config
>>> config.DISABLE_JIT =1

Make sure that the above is put on top of the script that you're running the
assignment from, before all other imports.
Importing numba and changing this variable after doing so yields rather confusing
errors.

For more details on other debug settings for Numba see https://numba.readthedocs.io/en/stable/reference/envvars.html.

When working with breakpoints in the assignment algorithms it is advantageous to have
access to
the assignment object in order to visualize the network or get additional information
that may not be available in the context where your breakpoint is set.
From the debugger one can always import the latest instantiated assignment object as
shown below:

>>> from dyntapy._context import running_assignment

The running_assignment object is either a `dyntapy.StaticAssignment` or a `dyntapy
.DynamicAssignment`, both share the `network` attribute.

>>> from dyntapy import show_network
>>> g = running_assignment.network
>>> show_network(g)

For more details on how to visualize link and node attributes do check the
documentation.



Adding Assignments
----------------------------------------


Instances of dyntapy's internal demand and supply objects, specified in `dyntapy.supply` and `dyntapy.demand`, are made available for both static and dynamic assignment instances.

If we have a `dyntapy.StaticAssignment` object given we can access the `internal_network` and `demand` attributes.

>>> from dyntapy import StaticAssignment
>>> assignment: Static Assignment
>>> def my_assignment_algorithm(network, demand):
>>>     tot_links = network.tot_links
>>>     free_flow_travel_times = network.links.length/network.links.free_speed
>>>     destination_nodes = demand.destinations
>>>     ...     

>>> my_assignment_algorithm(assignment.internal_network, assignment.demand)

Note that assignment algorithms that are implemented using the internal demand and supply objects can be accelerated using `Numba`_, which is not possible for generic python objects.

.. _Numba: https://numba.pydata.org/

You can always visualize the network and any link and node attributes that are generated during computations. 

Once your algorithm runs there is still some boilerplate code needed to fully integrate in dyntapy. For details do take a look at the existing assignments in `dyntapy.assignments`. The structure will essentially be the same for all static and dynamic assignments, respectively. Your compiled assignment routine returns all the arrays needed to fill the `dyntapy.results.StaticResults` or `dyntapy.results.DynamicResults`. The outer shell function creates the result object and returns it. 

Dynamic Assignments - Known Pitfalls
----------------------------------------

The dynamic assignment routines presented here work for the shown example(s) in the
tutorials, however they are not guaranteed to produce reliable result or fail
gracefully for any arbitrary network, travel demand and time configuration. If the
demand that you feed exceeds the local infrastructure and there is spillback into the
origin do not expect reasonable outputs. The same holds for demand that cannot leave
the network during the simulation,
there should always be some time periods in the dynamic assignment in which all the
queues can resolve and all vehicles can reach their destination. This
is very much following the principle of garbage-in-garbage-out.

It is best practice to first explore a small example and build some intuition for DTA
before moving on to more complex scenarios which slow you down because of their
larger computation time.

The complexity in DTA is mainly driven by the number of OD pairs, their intensity and
induced congestion, the
network's size
(in number of links and nodes) and the number of time steps. Ideally, your example to
experiment with should be low in complexity in all of those metrics.


