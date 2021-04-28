from . import libcrocoddyl_pywrap as _crocoddyl
from .deprecated import DeprecationHelper

ActivationModelSmoothAbs = DeprecationHelper(
    _crocoddyl.ActivationModelSmooth1Norm,
    "crocoddyl::ActivationModelSmoothAbs is deprecated: Use ActivationModelSmooth1Norm")

CostDataCentroidalMomentum = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataCentroidalMomentum is deprecated: Use CostDataResidual")

CostDataCoMPosition = DeprecationHelper(_crocoddyl.CostDataResidual,
                                        "crocoddyl::CostDataCoMPosition is deprecated: Use CostDataResidual")

CostDataContactCoPPosition = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataContactCoPPosition is deprecated: Use CostDataResidual")

CostDataContactForce = DeprecationHelper(_crocoddyl.CostDataResidual,
                                         "crocoddyl::CostDataContactForce is deprecated: Use CostDataResidual")

CostDataContactFrictionCone = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataContactFrictionCone is deprecated: Use CostDataResidual")

CostDataContactImpulse = DeprecationHelper(_crocoddyl.CostDataResidual,
                                           "crocoddyl::CostDataContactImpulse is deprecated: Use CostDataResidual")

CostDataContactWrenchCone = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataContactWrenchCone is deprecated: Use CostDataResidual")

CostDataControlGravContact = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataControlGravContact is deprecated: Use CostDataResidual")

CostDataControlGrav = DeprecationHelper(_crocoddyl.CostDataResidual,
                                        "crocoddyl::CostDataControlGrav is deprecated: Use CostDataResidual")

CostDataFramePlacement = DeprecationHelper(_crocoddyl.CostDataResidual,
                                           "crocoddyl::CostDataFramePlacement is deprecated: Use CostDataResidual")

CostDataFrameRotation = DeprecationHelper(_crocoddyl.CostDataResidual,
                                          "crocoddyl::CostDataFrameRotation is deprecated: Use CostDataResidual")

CostDataFrameTranslation = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataFrameTranslation is deprecated: Use CostDataResidual")

CostDataFrameVelocity = DeprecationHelper(_crocoddyl.CostDataResidual,
                                          "crocoddyl::CostDataFrameVelocity is deprecated: Use CostDataResidual")

CostDataImpulseCoM = DeprecationHelper(_crocoddyl.CostDataResidual,
                                       "crocoddyl::CostDataImpulseCoM is deprecated: Use CostDataResidual")

CostDataImpulseCoPPosition = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseCoPPosition is deprecated: Use CostDataResidual")

CostDataImpulseFrictionCone = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseFrictionCone is deprecated: Use CostDataResidual")

CostDataImpulseWrenchCone = DeprecationHelper(
    _crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseWrenchCone is deprecated: Use CostDataResidual")

CostDataState = DeprecationHelper(_crocoddyl.CostDataResidual,
                                  "crocoddyl::CostDataState is deprecated: Use CostDataResidual")

CostDataControl = DeprecationHelper(_crocoddyl.CostDataResidual,
                                    "crocoddyl::CostDataControl is deprecated: Use CostDataResidual")
