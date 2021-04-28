from . import libcrocoddyl_pywrap as _crocoddyl
from .deprecated import DeprecatedObject

ActivationModelSmoothAbs = DeprecatedObject(
    _crocoddyl.ActivationModelSmooth1Norm,
    "crocoddyl::ActivationModelSmoothAbs is deprecated: Use ActivationModelSmooth1Norm")

CostDataCentroidalMomentum = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataCentroidalMomentum is deprecated: Use CostDataResidual")

CostDataCoMPosition = DeprecatedObject(_crocoddyl.CostDataResidual,
                                       "crocoddyl::CostDataCoMPosition is deprecated: Use CostDataResidual")

CostDataContactCoPPosition = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataContactCoPPosition is deprecated: Use CostDataResidual")

CostDataContactForce = DeprecatedObject(_crocoddyl.CostDataResidual,
                                        "crocoddyl::CostDataContactForce is deprecated: Use CostDataResidual")

CostDataContactFrictionCone = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataContactFrictionCone is deprecated: Use CostDataResidual")

CostDataContactImpulse = DeprecatedObject(_crocoddyl.CostDataResidual,
                                          "crocoddyl::CostDataContactImpulse is deprecated: Use CostDataResidual")

CostDataContactWrenchCone = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataContactWrenchCone is deprecated: Use CostDataResidual")

CostDataControlGravContact = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataControlGravContact is deprecated: Use CostDataResidual")

CostDataControlGrav = DeprecatedObject(_crocoddyl.CostDataResidual,
                                       "crocoddyl::CostDataControlGrav is deprecated: Use CostDataResidual")

CostDataFramePlacement = DeprecatedObject(_crocoddyl.CostDataResidual,
                                          "crocoddyl::CostDataFramePlacement is deprecated: Use CostDataResidual")

CostDataFrameRotation = DeprecatedObject(_crocoddyl.CostDataResidual,
                                         "crocoddyl::CostDataFrameRotation is deprecated: Use CostDataResidual")

CostDataFrameTranslation = DeprecatedObject(_crocoddyl.CostDataResidual,
                                            "crocoddyl::CostDataFrameTranslation is deprecated: Use CostDataResidual")

CostDataFrameVelocity = DeprecatedObject(_crocoddyl.CostDataResidual,
                                         "crocoddyl::CostDataFrameVelocity is deprecated: Use CostDataResidual")

CostDataImpulseCoM = DeprecatedObject(_crocoddyl.CostDataResidual,
                                      "crocoddyl::CostDataImpulseCoM is deprecated: Use CostDataResidual")

CostDataImpulseCoPPosition = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseCoPPosition is deprecated: Use CostDataResidual")

CostDataImpulseFrictionCone = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseFrictionCone is deprecated: Use CostDataResidual")

CostDataImpulseWrenchCone = DeprecatedObject(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseWrenchCone is deprecated: Use CostDataResidual")

CostDataState = DeprecatedObject(_crocoddyl.CostDataResidual,
                                 "crocoddyl::CostDataState is deprecated: Use CostDataResidual")

CostDataControl = DeprecatedObject(_crocoddyl.CostDataResidual,
                                   "crocoddyl::CostDataControl is deprecated: Use CostDataResidual")
