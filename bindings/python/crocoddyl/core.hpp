///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_HPP_
#define PYTHON_CROCODDYL_CORE_HPP_

#include "python/crocoddyl/core/state-base.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/core/optctrl/shooting.hpp"
#include "python/crocoddyl/core/solver-base.hpp"
#include "python/crocoddyl/core/states/state-euclidean.hpp"
#include "python/crocoddyl/core/actions/unicycle.hpp"
#include "python/crocoddyl/core/actions/lqr.hpp"
#include "python/crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {
namespace python {

void exposeCore() {
  exposeStateAbstract();
  exposeActionAbstract();
  exposeDifferentialActionAbstract();
  exposeShootingProblem();
  exposeSolverAbstract();
  exposeStateEuclidean();
  exposeActionUnicycle();
  exposeActionLQR();
  exposeSolverDDP();
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_HPP_
