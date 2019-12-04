///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_HPP_

#include "python/crocoddyl/multibody/frames.hpp"
#include "python/crocoddyl/multibody/data/multibody.hpp"
#include "python/crocoddyl/multibody/data/multibody-in-contact.hpp"
#include "python/crocoddyl/multibody/data/multibody-in-impulse.hpp"
#include "python/crocoddyl/multibody/states/multibody.hpp"
#include "python/crocoddyl/multibody/actuations/floating-base.hpp"
#include "python/crocoddyl/multibody/actuations/full.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "python/crocoddyl/multibody/contact-base.hpp"
#include "python/crocoddyl/multibody/impulse-base.hpp"
#include "python/crocoddyl/multibody/costs/cost-sum.hpp"
#include "python/crocoddyl/multibody/costs/state.hpp"
#include "python/crocoddyl/multibody/costs/control.hpp"
#include "python/crocoddyl/multibody/costs/com-position.hpp"
#include "python/crocoddyl/multibody/costs/frame-placement.hpp"
#include "python/crocoddyl/multibody/costs/frame-translation.hpp"
#include "python/crocoddyl/multibody/costs/frame-rotation.hpp"
#include "python/crocoddyl/multibody/costs/frame-velocity.hpp"
#include "python/crocoddyl/multibody/costs/contact-force.hpp"
#include "python/crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "python/crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "python/crocoddyl/multibody/contacts/contact-3d.hpp"
#include "python/crocoddyl/multibody/contacts/contact-6d.hpp"
#include "python/crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "python/crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "python/crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "python/crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "python/crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "python/crocoddyl/multibody/actions/impulse-fwddyn.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeFrames();
  exposeDataCollectorMultibody();
  exposeDataCollectorMultibodyInContact();
  exposeDataCollectorMultibodyInImpulse();
  exposeStateMultibody();
  exposeActuationFloatingBase();
  exposeActuationFull();
  exposeCostMultibody();
  exposeContactAbstract();
  exposeImpulseAbstract();
  exposeCostSum();
  exposeCostState();
  exposeCostControl();
  exposeCostCoMPosition();
  exposeCostFramePlacement();
  exposeCostFrameTranslation();
  exposeCostFrameRotation();
  exposeCostFrameVelocity();
  exposeCostContactForce();
  exposeCostCentroidalMomentum();
  exposeContactMultiple();
  exposeContact3D();
  exposeContact6D();
  exposeImpulseMultiple();
  exposeImpulse3D();
  exposeImpulse6D();
  exposeDifferentialActionFreeFwdDynamics();
  exposeDifferentialActionContactFwdDynamics();
  exposeActionImpulseFwdDynamics();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_HPP_
