# Change log


## 0.2.4 (in development)
- support for link tolls in DialB
- Dial B reimplemented on the link-turn graph and destination based
- heuristic for finding k-shortest path with limited overlap


## 0.2.3 (08.09.2022)
- bugfix polygon creation 
- adding SUE assignment
- node highlighting for demand plots
- all show_* functions now optionally return the plot element for further processing 
- od_matrix_from_dataframes bugfix inconsistent column types
- debugging for static assignments
- docs for adding assignments 
- different scaling for link width in networks using euclidean coordinates vs web mercator
 
## 0.2.2 (27.06.2022)

- relabel_graph now optionally returns the node mapping as a dictionary
- bugfix: relabel_graph now retains the order of the centroids and intersection nodes by their ids.
- toy_network argument in visualization is replaced with euclidean
- remove result as argument for dynamic and static visualization
- places_around_place became a private function e.g. _places_around_place and moved to
dyntapy.demand_data
- added documentation on debugging
- bugfix dial_B for severely congested cases
- add function od_matrix_from_dataframes
- moved get_toy_network function to dyntapy.supply_data
- add_centroids_to_graph function has been renamed to add_centroids
- replace nx.MultiDiGraph with nx.DiGraph where applicable
- improving docs in all user-facing functions (dyntapy.function)
 
## 0.2.1 (02.03.2022)

- adding missing license file
- including changelog
- various minor bugfixes

## 0.2.0 (18.01.2022)

- major refactoring, reducing much of the namespace in the internals.
- extracting networks in different levels of coarsity with user-configurable buffers
- adding centroids based on place tags
- exposing all major functions in upper namespace (dyntapy.function)
- development of a generic result object for both Static and Dynamic Assignments
- Selected Link Analysis, for now only supported for Dial-B and SUN.
- Rework of the testing functions, support for CI/CD with pytest
- exposing shortest path computations to user in a simple API
- reworking notebooks including shortest path computations with 
 benchmarking against networkx and Selected Link Analysis
- some minor clean up in the visualization functions 
