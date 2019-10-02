///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_

#include "crocoddyl/core/solvers/box-ddp.hpp"
#include "crocoddyl/core/solvers/box_qp.h"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeSolverBoxDDP() {
  bp::class_<SolverBoxDDP, bp::bases<SolverDDP> >(
      "SolverBoxDDP",
      "Box-constrained DDP solver.\n\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<ShootingProblem&>(bp::args(" self", " problem"),
                                 "Initialize the vector dimension.\n\n"
                                 ":param problem: shooting problem.")[bp::with_custodian_and_ward<1, 2>()]);
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
