///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeSolverBoxFDDP() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverBoxFDDP>>();

  bp::class_<SolverBoxFDDP, bp::bases<SolverFDDP>>(
      "SolverBoxFDDP",
      "Box-constrained FDDP solver.\n\n"
      ":param shootingProblem: shooting problem (list of action models along "
      "trajectory.)",
      bp::init<boost::shared_ptr<ShootingProblem>>(
          bp::args("self", "problem"), "Initialize the vector dimension.\n\n"
                                       ":param problem: shooting problem."))
      .add_property("Quu_inv",
                    make_function(&SolverBoxFDDP::get_Quu_inv,
                                  bp::return_internal_reference<>()),
                    "inverse of the Quu computed by the box QP");
}

} // namespace python
} // namespace crocoddyl
