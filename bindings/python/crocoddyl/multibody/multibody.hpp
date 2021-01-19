///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_

#include <pinocchio/fwd.hpp>
#include "python/crocoddyl/fwd.hpp"

namespace crocoddyl {
namespace python {

void exposeFrames();
void exposeFrictionCone();
void exposeWrenchCone();
void exposeStateMultibody();
void exposeActuationFloatingBase();
void exposeActuationFull();
void exposeActuationModelMultiCopterBase();
void exposeContactAbstract();
void exposeImpulseAbstract();
void exposeContactMultiple();
void exposeImpulseMultiple();
void exposeDataCollectorMultibody();
void exposeDataCollectorContacts();
void exposeDataCollectorImpulses();
void exposeDifferentialActionFreeFwdDynamics();
void exposeDifferentialActionContactFwdDynamics();
void exposeActionImpulseFwdDynamics();
void exposeCostState();
void exposeCostControlGrav();
void exposeCostCoMPosition();
void exposeCostCentroidalMomentum();
void exposeCostFramePlacement();
void exposeCostFrameTranslation();
void exposeCostFrameRotation();
void exposeCostFrameVelocity();
void exposeCostContactForce();
void exposeCostContactWrenchCone();
void exposeCostContactCoPPosition();
void exposeCostContactFrictionCone();
void exposeCostContactImpulse();
void exposeCostImpulseFrictionCone();
void exposeCostImpulseWrenchCone();
void exposeCostImpulseCoPPosition();
void exposeCostImpulseCoM();
void exposeCostPairCollisions();
void exposeContact2D();
void exposeContact3D();
void exposeContact6D();
void exposeImpulse3D();
void exposeImpulse6D();

void exposeMultibody();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_
