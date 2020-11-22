#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import numpy as np
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import logging
import stapy.settings
import inspect
import os
import datetime
from functools import wraps
import time


def __create_green_to_red_cm(color_format: str):
    """

    Parameters
    ----------
    color_format : either 'hex' or 'rgba' to get the map in different formats

    Returns colormap as list
    -------

    """
    x = np.linspace(0, 1, 256)
    rgba = cm.get_cmap('hsv', 256)(x)
    # which element in rgba is the most green and the least red and blue
    temp = np.copy(rgba)
    np.negative(temp[:, 1:2], temp[:, 1:2])  # negative sign for green --> minimizing rowsum corresponds to green
    index_green = temp.sum(axis=1).argmin()
    rgba = np.flipud(rgba[:index_green, :])  # inverse the resulting colormap to get (green --> yellow--> red)
    rgba = np.insert(rgba, 0, np.array([0.5, 0.5, 0.5, 1]), axis=0)  # adding grey as baseline color for no load
    if color_format == 'hex':
        return [to_hex(color, keep_alpha=False) for color in rgba]
    else:
        return rgba


def timeit(my_func):
    #simple decorators that allows to time functions calls
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed


def _name_test(data, key: str, value=None):
    """
    help function to check for used keys in nx graphs

    examples:
    list = [(u,v,data) for u,v,data in g.edges.data() if name_test(data,'key', 'value']
    list will then contain all edges for which data contains 'key':'value'
    Parameters
    ----------
    data :
    key :
    value :

    Returns
    -------

    """
    try:
        if value is not None:
            local_bool = data[f'{key}'] == f'{value}'
        else:
            data[f'{key}'] = data[f'{key}']  # simply accessing element to possibly trigger KeyError
            local_bool = True  # no KeyError triggered, element has value for key
    except KeyError:
        local_bool = False
    return local_bool


def log(message, level=None, print=False):
    """
    Record a message in the log file or/and print to the console

    Parameters
    ----------
    message : string
    level : int
        logger level
    Returns
    -------
    None
    """
    if level is None:
        level = stapy.settings.log_level
    if stapy.settings.log_to_file:
        # create a new logger with the calling script's name, or access the existing one
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        logger = get_logger(mod.__name__)
        if level == logging.DEBUG:
            logger.debug(message)
        elif level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
    if print:
        print(message)


def get_logger(name):
    """
    Create a logger or return the current one if already instantiated.

    Parameters
    ----------
    level : int
        one of the logger.level constants
    name : string
        name of the logger
    filename : string
        name of the log file

    Returns
    -------
    logger.logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    filename = stapy.settings.log_filename

    # if a logger with this name is not already set up
    if not getattr(logger, 'handler_set', None):

        # get today's date and construct a log filename
        todays_date = datetime.datetime.today().strftime('%Y_%m_%d')
        log_filename = os.path.join(stapy.settings.log_folder, '{}_{}.log'.format(filename, todays_date))

        # if the logs folder does not already exist, create it
        if not os.path.exists(stapy.settings.log_folder):
            os.makedirs(stapy.settings.log_folder)

        # create file handler and log formatter and set them up
        handler = logging.FileHandler(log_filename, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s %(levelname)s @%(name)s.py: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.handler_set = True

    return logger


def np_to_py_type_conversion(value):
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
