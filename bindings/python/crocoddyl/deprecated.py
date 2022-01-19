import functools
import warnings


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


class DeprecationHelper(object):

    def __init__(self, new_target, old_name):
        self.new_target = new_target
        self.warning_str = '%s is deprecated: Use %s' % (old_name, new_target.__name__)

    def _warn(self):
        warnings.warn(self.warning_str)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)
