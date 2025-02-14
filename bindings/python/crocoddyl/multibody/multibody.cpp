///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeFrictionCone();
  exposeWrenchCone();
  exposeCoPSupport();
  exposeStateMultibody();
  exposeActuationFloatingBase();
  exposeActuationFull();
  exposeActuationFloatingBaseThruster();
  exposeForceAbstract();
  exposeContactAbstract();
  exposeImpulseAbstract();
  exposeContactMultiple();
  exposeImpulseMultiple();
  exposeDataCollectorMultibody();
  exposeDataCollectorContacts();
  exposeDataCollectorImpulses();
  exposeDifferentialActionFreeFwdDynamics();
  exposeDifferentialActionFreeInvDynamics();
  exposeDifferentialActionContactFwdDynamics();
  exposeDifferentialActionContactInvDynamics();
  exposeActionImpulseFwdDynamics();
  exposeResidualState();
  exposeResidualCentroidalMomentum();
  exposeResidualCoMPosition();
  exposeResidualContactForce();
  exposeResidualContactFrictionCone();
  exposeResidualContactCoPPosition();
  exposeResidualContactWrenchCone();
  exposeResidualContactControlGrav();
  exposeResidualControlGrav();
  exposeResidualFramePlacement();
  exposeResidualFrameRotation();
  exposeResidualFrameTranslation();
  exposeResidualFrameVelocity();
  exposeResidualImpulseCoM();

#ifdef PINOCCHIO_WITH_HPP_FCL
#ifdef CROCODDYL_WITH_PAIR_COLLISION
  exposeResidualPairCollision();
#endif
#endif

  exposeContact1D();
  exposeContact2D();
  exposeContact3D();
  exposeContact6D();
  exposeImpulse3D();
  exposeImpulse6D();
}

}  // namespace python
}  // namespace crocoddyl
