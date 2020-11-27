///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
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
  exposeStateMultibody();
  exposeActuationFloatingBase();
  exposeActuationFull();
  exposeActuationModelMultiCopterBase();
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
  exposeContact2D();
  exposeContact3D();
  exposeContact6D();
  exposeImpulse3D();
  exposeImpulse6D();
}

}  // namespace python
}  // namespace crocoddyl
