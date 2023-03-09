import logging
logging.basicConfig(level=logging.DEBUG)

from numba import config
config.DEBUG = 1
# config.DEBUG_TYPEINFER = 1
# config.DEBUGINFO = 1



from test_dyntapy import  test_dial_b

test_dial_b()