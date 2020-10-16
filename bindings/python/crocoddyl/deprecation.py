from . import libcrocoddyl_pywrap as _crocoddyl
from .deprecated import deprecated


class ActivationModelSmoothAbs(_crocoddyl.ActivationModelSmooth1Norm):
    @deprecated("crocoddyl::ActivationModelSmoothAbs is deprecated: Use ActivationModelSmooth1Norm")
    def __init__(self, nr, eps=1.):
        _crocoddyl.ActivationModelSmooth1Norm.__init__(self, nr, eps)
