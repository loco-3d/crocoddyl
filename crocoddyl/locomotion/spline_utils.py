import numpy as np

def polyfitND(x, y, deg, eps, rcond=None, full=False, w=None, cov=False):
  """ Return the polynomial fitting (and its derivative)
  for a set of points x, y where y=p(x).
  This is a wrapper of np.polyfit for more than one dimensions.
  
  :params x: Points where polynomial is being computed. e.g. on a timeline.
  :params y: value of the curve at x, y=p(x). each row corresponds to a point.
  :params deg: degree of the polynomial being computed.
  :params eps: tolerance to confirm the fidelity of the polynomial fitting.
  :params rcond: condition number of the fit. Default None
  :params full: True to return the svd. False to return just coeffs. Default False.
  :params w: weights for y coordinates. Default None.
  :params cov:estimate and covariance of the estimate. Default False.

  :returns array p where each row corresponds to the coeffs of a dimension in format np.poly1d
  """
  assert len(x.shape)==1; #x is an array.
  p = np.array([None,]*y.shape[1])
  dp = np.array([None,]*y.shape[1])
  for i,y_a in enumerate(y.T):
    p_a , residual, _, _, _ = np.polyfit(x, np.asarray(y_a).squeeze(),
                                         deg, rcond, full, w, cov);
    p[i] = np.poly1d(p_a[:])
    dp[i] = np.poly1d(p_a[:]).deriv(1)
    assert(residual <=eps)
  return p, dp


def findDuplicates(arr, tol=1e-8):
    """ Takes a sorted array and returns the index of duplicated valuesinside the array
    
    :params arr: np.array. e.g., a timeline
    :params tol. optional. criterion to know if values are being duplicated.
    :returns list of duplicated indices
    """
    duplicates = []
    for i, a in enumerate(arr):
        if i==0:
            pass
        else:
            if abs(a-arr[i-1])< tol:
                duplicates.append(i)
    return duplicates

def removeDuplicates(arr, indices):
    """ Takes an arr and a sorted list of indices.
    Removes arr[i] for i in indices and returns a new array.
    if arr is 2D, removes the i^{th} row for i in indices.

    :params arr: np.array. e.g., a timeline
    :params indices: sorted list of indices to remove.
    :returns new_arr: np.array
    """

    for i in indices[::-1]:
        if len(arr.shape)==1:
            arr = np.delete(arr,i)
        elif len(arr.shape)==2:
            arr = np.delete(arr, i, axis=0)
    return arr
