import numpy as np

EPS = np.finfo(float).eps
NUMDIFF_MODIFIER = 3e4


class NumDiffException(Exception):
    """Raised when the NumDiff values are too high"""

    pass


def assertNumDiff(A, B, threshold):
    """Assert analytical derivatives against NumDiff using the error norm.

    :param A: analytical derivatives
    :param B: NumDiff derivatives
    :param threshold: absolute tolerance
    """
    if not np.allclose(A, B, atol=threshold):
        value = np.linalg.norm(A - B)
        raise NumDiffException(
            f"NumDiff exception, with residual of {value:.4g}, above threshold "
            f"{threshold:.4g}"
        )
