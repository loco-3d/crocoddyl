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
#include "python/crocoddyl/multibody/states/state-multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeMultibody() {
  exposeCostMultibody();
  exposeStateMultibody();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_HPP_