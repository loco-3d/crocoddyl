///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PYTHON_CORE_HPP_
#define CROCODDYL_PYTHON_CORE_HPP_

#include <crocoddyl/python/core/state-base.hpp>
#include <crocoddyl/python/core/action-base.hpp>
#include <crocoddyl/python/core/states/state-euclidean.hpp>
#include <crocoddyl/python/core/actions/unicycle.hpp>
#include <crocoddyl/python/core/actions/lqr.hpp>

namespace crocoddyl {
namespace python {

void exposeCore() {
  exposeStateAbstract();
  exposeActionAbstract();
  exposeStateEuclidean();
  exposeActionUnicycle();
  exposeActionLQR();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // CROCODDYL_PYTHON_CORE_HPP_
