### Class Structure and Design Choices
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
    List <Tuple> event_queue
    Array1D <float64> controlled
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