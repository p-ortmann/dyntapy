# Change log
## 0.2.2 (in progress)

- remove result as argument for dynamic and static visualization
- places_around_place became a private function e.g. _places_around_place and moved to
dyntapy.demand_data
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
