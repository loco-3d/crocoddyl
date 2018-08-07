import numpy as np


class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def isPositiveDefinitive(A):
  """ Checks if the matrix is positive definitive.

  :param A matrix
  """
  try:
    _ = np.linalg.cholesky(A)
  except np.linalg.LinAlgError:
    print "Matrix is not positive definitive"
    return False
  return True


def plot(x, y, yerr=None, color=None, alpha_fill=0.3, ax=None):
  import matplotlib.pyplot as plt
  ax = ax if ax is not None else plt.gca()
  if color is None:
    color = ax._get_lines.color_cycle.next()
  ax.plot(x, y, color=color)
  if yerr is not None:
    if np.isscalar(yerr) or len(yerr) == len(y):
      ymin = y - yerr
      ymax = y + yerr
    elif len(yerr) == 2:
      ymin, ymax = yerr
    ax.fill_between(x,
      ymax.reshape(-1), ymin.reshape(-1), color=color, alpha=alpha_fill)


def show_plot():
  import matplotlib.pyplot as plt
  plt.show()

def assertClass(derivated, abstract):
  import inspect
  assert inspect.getmro(derivated.__class__)[-2].__name__ == abstract, \
      'The ' + derivated.__class__.__name__ + ' class has to derived from the ' +\
      abstract + ' abstract class.'