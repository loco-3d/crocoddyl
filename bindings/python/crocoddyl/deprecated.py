import functools
import warnings
from types import FunctionType


class DeprecatedWarning(UserWarning):
    pass


def deprecated(instructions):
    """Flags a method as deprecated.
    Args:
        instructions: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """
    def decorator(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = 'Call to deprecated function {}. {}'.format(func.__name__, instructions)
            warnings.warn(message, category=DeprecatedWarning, stacklevel=2)
            return func(*args, **kwargs)

        instructions_doc = 'Deprecated: ' + instructions
        if wrapper.__doc__ is None:
            wrapper.__doc__ = instructions_doc
        else:
            wrapper.__doc__ = wrapper.__doc__.rstrip() + '\n\n' + instructions_doc
        return wrapper

    return decorator


def depr_wrapper(f, warning):
    def new(*args, **kwargs):
        if not args[0].warned:
            print("Deprecated Warning: %s" % warning)
            args[0].warned = True
        return f(*args, **kwargs)

    return new


def DeprecatedObject(o, warning):
    class temp(o):
        pass

    temp.__name__ = "Deprecated_%s" % o.__class__.__name__
    output = temp.__new__(temp, o)

    output.warned = True
    wrappable_types = (type(int.__add__), type(zip), FunctionType)
    unwrappable_names = ("__str__", "__unicode__", "__repr__", "__getattribute__", "__setattr__")

    for method_name in dir(temp):
        if not type(getattr(temp, method_name)) in wrappable_types:
            continue
        if method_name in unwrappable_names:
            continue
        if method_name != "__class__":
            setattr(temp, method_name, depr_wrapper(getattr(temp, method_name), warning))

    output.warned = False
    return output
