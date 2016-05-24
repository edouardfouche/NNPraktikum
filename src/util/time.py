# -*- coding: utf-8 -*-


import os
from time import time
from functools import wraps

"""
Time measure
"""

def timed(function):
    """
    Measure the elapsed time of custom functions of the package. Should be used 
    as a decorator.
    """
    @wraps(function)
    def wrapper(*args, **kwds):
        """Calculate elapsed time in seconds"""
        start = time()
        result = function(*args, **kwds)
        elapsed = time() - start
        if os.environ['VERBOSE'] == 'True':
            print("Execution time: %s seconds." %elapsed)
        return result
    return wrapper