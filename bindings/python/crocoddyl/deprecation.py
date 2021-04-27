from . import libcrocoddyl_pywrap as _crocoddyl
from .deprecated import deprecated


class ActivationModelSmoothAbs(_crocoddyl.ActivationModelSmooth1Norm):
    @deprecated("crocoddyl::ActivationModelSmoothAbs is deprecated: Use ActivationModelSmooth1Norm")
    def __init__(self, nr, eps=1.):
        _crocoddyl.ActivationModelSmooth1Norm.__init__(self, nr, eps)

#@deprecated("crocoddyl::CostDataCentroidalMomentum is deprecated: Use CostDataResidual")
CostDataCentroidalMomentum = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataCoMPosition is deprecated: Use CostDataResidual")
CostDataCoMPosition = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataContactCoPPosition is deprecated: Use CostDataResidual")
CostDataContactCoPPosition = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataContactForce is deprecated: Use CostDataResidual")
CostDataContactForce = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataContactFrictionCone is deprecated: Use CostDataResidual")
CostDataContactFrictionCone = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataContactImpulse is deprecated: Use CostDataResidual")
CostDataContactImpulse = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataContactWrenchCone is deprecated: Use CostDataResidual")
CostDataContactWrenchCone = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataControlGravContact is deprecated: Use CostDataResidual")
CostDataControlGravContact = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataControlGrav is deprecated: Use CostDataResidual")
CostDataControlGrav = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataFramePlacement is deprecated: Use CostDataResidual")
CostDataFramePlacement = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataFrameRotation is deprecated: Use CostDataResidual")
CostDataFrameRotation = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataFrameTranslation is deprecated: Use CostDataResidual")
CostDataFrameTranslation = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataFrameVelocity is deprecated: Use CostDataResidual")
CostDataFrameVelocity = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataImpulseCoM is deprecated: Use CostDataResidual")
CostDataImpulseCoM = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataImpulseCoPPosition is deprecated: Use CostDataResidual")
CostDataImpulseCoPPosition = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataImpulseFrictionCone is deprecated: Use CostDataResidual")
CostDataImpulseFrictionCone = _crocoddyl.CostDataResidual

#@deprecated("crocoddyl::CostDataImpulseWrenchCone is deprecated: Use CostDataResidual")
CostDataImpulseWrenchCone = _crocoddyl.CostDataResidual 

#@deprecated("crocoddyl::CostDataState is deprecated: Use CostDataResidual")
CostDataState = _crocoddyl.CostDataResidual 

#@deprecated("crocoddyl::CostDataControl is deprecated: Use CostDataResidual")
CostDataControl = _crocoddyl.CostDataResidual 

