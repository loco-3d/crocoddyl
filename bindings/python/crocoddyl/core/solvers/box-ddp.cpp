///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/solvers/box-ddp.hpp"

namespace crocoddyl {
namespace python {

void exposeSolverBoxDDP() {
  bp::class_<SolverBoxDDP, bp::bases<SolverDDP> >(
      "SolverBoxDDP",
      "Box-constrained DDP solver.\n\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<boost::shared_ptr<ShootingProblem> >(bp::args("self", "problem"),
                                                    "Initialize the vector dimension.\n\n"
                                                    ":param problem: shooting problem."))
      .add_property("Quu_inv",
                    make_function(&SolverBoxDDP::get_Quu_inv, bp::return_internal_reference<>()),
                    "inverse of the Quu computed by the box QP");
}

}  // namespace python
}  // namespace crocoddyl
