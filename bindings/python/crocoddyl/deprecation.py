from . import libcrocoddyl_pywrap as _crocoddyl
from .deprecated import DeprecationHelper

ActivationModelSmoothAbs = DeprecationHelper(_crocoddyl.ActivationModelSmooth1Norm,
                                             "crocoddyl::ActivationModelSmoothAbs")

CostDataCentroidalMomentum = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataCentroidalMomentum")

CostDataCoMPosition = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataCoMPosition")

CostDataContactCoPPosition = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataContactCoPPosition")

CostDataContactForce = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataContactForce")

CostDataContactFrictionCone = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataContactFrictionCone")

CostDataContactImpulse = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataContactImpulse")

CostDataContactWrenchCone = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataContactWrenchCone")

CostDataControlGravContact = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataControlGravContact")

CostDataControlGrav = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataControlGrav")

CostDataFramePlacement = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataFramePlacement")

CostDataFrameRotation = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataFrameRotation")

CostDataFrameTranslation = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataFrameTranslation")

CostDataFrameVelocity = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataFrameVelocity")

CostDataImpulseCoM = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseCoM")

CostDataImpulseCoPPosition = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseCoPPosition")

CostDataImpulseFrictionCone = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseFrictionCone")

CostDataImpulseWrenchCone = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataImpulseWrenchCone")

CostDataState = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataState")

CostDataControl = DeprecationHelper(_crocoddyl.CostDataResidual, "crocoddyl::CostDataControl")
