# import logging
# logging.basicConfig(level=logging.DEBUG)

# from numba import config
# config.DEBUG = 1
# config.DEBUG_TYPEINFER = 1
# config.DEBUGINFO = 1



from test_dyntapy import  test_dial_b, test_get_graph, test_get_toy_networks, \
    test_node_model, test_sun, test_selected_link_analysis, test_dta, test_kspwlo

# test_get_toy_networks()
# test_get_graph()
# test_node_model()
# test_sun()
# test_kspwlo()
test_dial_b()
test_selected_link_analysis()
test_dta()