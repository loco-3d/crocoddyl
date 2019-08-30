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
#include "python/crocoddyl/multibody/costs/frame-velocity.hpp"
#include "python/crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "python/crocoddyl/multibody/contacts/contact-3d.hpp"
#include "python/crocoddyl/multibody/contacts/contact-6d.hpp"
#include "python/crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "python/crocoddyl/multibody/actions/contact-fwddyn.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeFrames();
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
  exposeCostFrameVelocity();
  exposeContactMultiple();
  exposeContact3D();
  exposeContact6D();
  exposeDifferentialActionFreeFwdDynamics();
  exposeDifferentialActionContactFwdDynamics();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_HPP_
