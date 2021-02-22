///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeFrames();
  exposeFrictionCone();
  exposeWrenchCone();
  exposeCoPSupport();
  exposeStateMultibody();
  exposeActuationFloatingBase();
  exposeActuationFull();
  exposeActuationModelMultiCopterBase();
  exposeForceAbstract();
  exposeContactAbstract();
  exposeImpulseAbstract();
  exposeContactMultiple();
  exposeImpulseMultiple();
  exposeDataCollectorMultibody();
  exposeDataCollectorContacts();
  exposeDataCollectorImpulses();
  exposeDifferentialActionFreeFwdDynamics();
  exposeDifferentialActionContactFwdDynamics();
  exposeActionImpulseFwdDynamics();
  exposeResidualState();
  exposeResidualCentroidalMomentum();
  exposeResidualCoMPosition();
  exposeResidualContactForce();
  exposeResidualContactFrictionCone();
  exposeResidualContactCoPPosition();
  exposeResidualContactWrenchCone();
  exposeResidualControlGrav();
  exposeResidualFramePlacement();
  exposeResidualFrameRotation();
  exposeResidualFrameTranslation();
  exposeResidualFrameVelocity();
  exposeResidualImpulseCoM();
  exposeCostState();
  exposeCostCoMPosition();
  exposeCostControlGrav();
  exposeCostControlGravContact();
  exposeCostCentroidalMomentum();
  exposeCostFramePlacement();
  exposeCostFrameTranslation();
  exposeCostFrameRotation();
  exposeCostFrameVelocity();
  exposeCostContactForce();
  exposeCostContactWrenchCone();
  exposeCostContactImpulse();
  exposeCostContactCoPPosition();
  exposeCostContactFrictionCone();
  exposeCostImpulseCoM();
  exposeCostImpulseFrictionCone();
  exposeCostImpulseWrenchCone();
  exposeCostImpulseCoPPosition();
  exposeResidualControlGrav();
  exposeContact2D();
  exposeContact3D();
  exposeContact6D();
  exposeImpulse3D();
  exposeImpulse6D();
}

}  // namespace python
}  // namespace crocoddyl
