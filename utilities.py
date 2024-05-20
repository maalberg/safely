import numpy as np
from functools import wraps

def stack_args(first: int = 0):
    """
    Stack the arguments of a function starting from ``first`` argument.
    The arguments are stacked horizontally, i.e. as columns
    of an nd-array. Note that this method filters out
    none arguments; this behavior is experimental.
    """
    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # filter out none arguments and make sure the arguments are formatted as 2d arrays
            stack = tuple(filter(lambda arg: True if arg is not None else False, args[first:]))
            stack = tuple(map(np.atleast_2d, stack))

            args_n = len(stack)
            if args_n == 1:
                args_stacked = stack
            else:
                args_stacked = (np.column_stack(stack),)

            # update the list of arguments
            args = args[:first] + args_stacked
            return func(*args, **kwargs)
        return wrapped_func
    return wrap
