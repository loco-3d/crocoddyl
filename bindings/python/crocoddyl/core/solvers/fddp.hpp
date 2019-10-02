///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_FDDP_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_FDDP_HPP_

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeSolverFDDP() {
  bp::class_<SolverFDDP, bp::bases<SolverDDP> >(
      "SolverFDDP",
      "Feasibility-prone DDP (FDDP) solver.\n\n"
      "The FDDP solver computes an optimal trajectory and control commands by iterates\n"
      "running backward and forward passes. The backward-pass updates locally the\n"
      "quadratic approximation of the problem and computes descent direction,\n"
      "and the forward-pass rollouts this new policy by integrating the system dynamics\n"
      "along a tuple of optimized control commands U*.\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<ShootingProblem&>(bp::args(" self", " problem"),
                                 "Initialize the vector dimension.\n\n"
                                 ":param problem: shooting problem.")[bp::with_custodian_and_ward<1, 2>()])
      .def("expectedImprovement", &SolverFDDP::expectedImprovement,
           bp::return_value_policy<bp::copy_const_reference>(), bp::args(" self"),
           "Return two scalars denoting the quadratic improvement model\n\n"
           "For computing the expected improvement, you need to compute first\n"
           "the search direction by running computeDirection. The quadratic\n"
           "improvement model is described as dV = f_0 - f_+ = d1*a + d2*a**2/2.\n"
           "Additionally, you need to update the expected model by running\n"
           "updateExpectedImprovement.")
      .def("updateExpectedImprovement", &SolverFDDP::updateExpectedImprovement,
           bp::return_value_policy<bp::copy_const_reference>(), bp::args(" self"),
           "Update the expected improvement model\n\n")
      .add_property("th_acceptNegStep", bp::make_function(&SolverFDDP::get_th_acceptnegstep),
                    bp::make_function(&SolverFDDP::set_th_acceptnegstep),
                    "threshold for step acceptance in ascent direction");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_FDDP_HPP_
