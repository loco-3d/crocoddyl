///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_

#include <pinocchio/fwd.hpp>

#include "python/crocoddyl/fwd.hpp"

namespace crocoddyl {
namespace python {

void exposeFrictionCone();
void exposeWrenchCone();
void exposeCoPSupport();
void exposeStateMultibody();
void exposeActuationFloatingBase();
void exposeActuationFull();
void exposeActuationFloatingBaseThruster();
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
void exposeDifferentialActionFreeInvDynamics();
void exposeDifferentialActionContactFwdDynamics();
void exposeDifferentialActionContactInvDynamics();
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

void exposeContact1D();
void exposeContact2D();
void exposeContact3D();
void exposeContact6D();
void exposeContact6DLoop();
void exposeImpulse3D();
void exposeImpulse6D();
void exposeMultibody();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_MULTIBODY_HPP_
