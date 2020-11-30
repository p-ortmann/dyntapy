### Class Structure and Design Choices
Below you can see a schematic outlining the current implementation of the data structures that are used in the class 
definitions. CSRMatrices are used extensively to minimize access times to different node-, link-, or turn based 
attributes, for further explanations consult the comments in the files.

We want to point out here the purpose and intention behind keeping track of DynamicEvent objects that are associated 
with the Network. 
First, some background: Throughout the day attributes that we,as modellers, associate with entities may vary. 
This can happen in two ways, (i) deterministically: Think about reductions in speed for roads in the vicinity of schools, 
lane reversals or different traffic signal plans for off and on peak hours. And (ii) dynamically if we have 
controllers in play that react to the network state __during__ the simulation.
This has some implications for the design of our data structures. We want to avoid storing large matrices with
attributes duplicated for different time intervals when we know that most of the time these attributes will most likely not change. 

'''mermaid
classDiagram
    Network -- Links
    Network -- Nodes
    Network -- Turns
    Network : Links
    Network : Nodes
    Network : Turns
    Network : List <DynamicEvent> dynamic_events
    Network : List <CSRMatrix> static_events
    Network : process_events(time)
    Network : get_controller_response(time, results)
    Links: Array1D <float64> capacity
    Links: Array1D <float64> kJam
    Links: Array1D <float64> sending_flow
    Links: Array1D <float64> receiving_flow
    Links: Array1D <float64> length
    Links: Array1D <int64> from_node
    Links: Array1D <int64> to_node
    Links: Array3D <float64> travel_time
    Links: Array3D <float64> flows
    Links: CSRMatrix <int64> forward
    Links: CSRMatrix <int64> backward
    Links: CSRMatrix <float64> event_changes
    Links: ...
    class DynamicEvent{
    List <Tuple> __event_queue
    Array1D <float64> __control_array
    add_event(time, obj_index, val)
    get_next_event()
    pop_next_event()
    }
    class Nodes{
    CSRMatrix <int64> forward
    CSRMatrix <int64> backward
    CSRMatrix <float64> event_changes
    ...
    }
    class Turns{
    CSRMatrix <float64> fractions
    CSRMatrix <bool> db_restrictions
    CSRMatrix <float64> event_changes
    ...
    }
'''
