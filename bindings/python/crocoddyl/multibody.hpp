///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_HPP_

#include "python/crocoddyl/multibody/cost-base.hpp"
#include "python/crocoddyl/multibody/states/multibody.hpp"
#include "python/crocoddyl/multibody/costs/cost-sum.hpp"
#include "python/crocoddyl/multibody/costs/state.hpp"
#include "python/crocoddyl/multibody/costs/control.hpp"
#include "python/crocoddyl/multibody/costs/frame-placement.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeCostMultibody();
  exposeStateMultibody();
  exposeCostSum();
  exposeCostState();
  exposeCostControl();
  exposeCostFramePlacement();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_HPP_