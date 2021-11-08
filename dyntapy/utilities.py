#  This file is part of the Traffic Assignment Package developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.mech.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
#
#
import numpy as np
from matplotlib.colors import to_hex, to_rgba
import matplotlib.cm as cm
import logging
import dyntapy.settings as settings
import inspect
import os
import datetime
from functools import wraps
import time
from numba import objmode, njit
from dyntapy.settings import log_numba, dynamic_parameters
from dyntapy.__init__ import data_folder

default_link_color=dynamic_parameters.visualization.link_color


cleared_log = False


def __create_green_to_red_cm():
    """

    Returns colormap as hex list
    -------

    """
    x = np.linspace(0, 1, 256)
    rgba = cm.get_cmap('hsv', 256)(x)
    # which element in rgba is the most green and the least red and blue
    temp = np.copy(rgba)
    np.negative(temp[:, 1:2], temp[:, 1:2])  # negative sign for green --> minimizing rowsum corresponds to green
    index_green = temp.sum(axis=1).argmin()
    rgba = np.flipud(rgba[:index_green, :])  # inverse the resulting colormap to get (green --> yellow--> red)
    new_cm = [to_hex(color, keep_alpha=False) for color in rgba]
    new_cm.insert(0, default_link_color)
    return new_cm


def timeit(my_func):
    # simple decorators that allows to time functions calls
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed


@njit(cache=True)
def _log(message, level=settings.log_level,to_console=False):
    """

    Parameters
    ----------
    message : string to log
    level: int, see py docs
    to_console: whether to print

    Returns
    -------

    """
    if to_console:
        print(message)
    if log_numba:
        with objmode():
            print('entered objmode..')
            log(message,level=level)
    else:
        pass


def log(message, level=None, to_console=False):
    """
    Record a message in the log file or/and print to the console

    Parameters
    ----------
    to_console : bool
    message : string
    level : int
        logger level
    Returns
    -------
    None
    """
    if level is None:
        level = settings.log_level
    if settings.log_to_file:
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
    if to_console:
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

    # if a logger with this name is not already set up
    if not getattr(logger, 'handler_set', None):

        # get today's date and construct a log filename
        todays_date = datetime.datetime.today().strftime('%Y_%m_%d')
        log_filename = os.path.join(settings.log_folder, '{}_{}.log'.format('dtapy', todays_date))
        global cleared_log
        if not cleared_log:
            try:
                open(log_filename, 'w').close()  # clears the log from previous runs
            except FileNotFoundError:
                pass
            cleared_log = True

        # if the logs folder does not already exist, create it
        if not os.path.exists(settings.log_folder):
            os.makedirs(settings.log_folder)

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


def _filepath(name: str, check_path_valid=False):
    assert isinstance(name, str)
    file_path = os.path.join(data_folder, str(name.lower() + '.pickle'))
    if check_path_valid:
        if not os.path.isfile(file_path):
            raise NameError(f'{name}.pickle not found in data folder!')
    return file_path
