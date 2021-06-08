///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
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
void exposeCoPSupport();
void exposeStateMultibody();
void exposeActuationFloatingBase();
void exposeActuationFull();
void exposeActuationModelMultiCopterBase();
void exposeForceAbstract();
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
void exposeResidualState();
void exposeResidualCentroidalMomentum();
void exposeResidualCoMPosition();
void exposeResidualContactForce();
void exposeResidualContactFrictionCone();
void exposeResidualContactCoPPosition();
void exposeResidualContactWrenchCone();
void exposeResidualContactControlGrav();
void exposeResidualControlGrav();
void exposeResidualFramePlacement();
void exposeResidualFrameRotation();
void exposeResidualFrameTranslation();
void exposeResidualFrameVelocity();
void exposeResidualImpulseCoM();

#ifdef PINOCCHIO_WITH_HPP_FCL
void exposeResidualPairCollision();
#endif

void exposeCostState();
void exposeCostCoMPosition();
void exposeCostControlGrav();
void exposeCostControlGravContact();
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
void exposeContact2D();
void exposeContact3D();
void exposeContact6D();
void exposeImpulse3D();
void exposeImpulse6D();

void exposeMultibody();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_
