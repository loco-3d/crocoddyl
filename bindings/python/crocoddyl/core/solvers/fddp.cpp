///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverFDDP_solves, SolverFDDP::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverFDDP_computeDirections, SolverDDP::computeDirection, 0, 1)

void exposeSolverFDDP() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverFDDP> >();

  bp::class_<SolverFDDP, bp::bases<SolverDDP> >(
      "SolverFDDP",
      "Feasibility-driven DDP (FDDP) solver.\n\n"
      "The FDDP solver computes an optimal trajectory and control commands by iterates\n"
      "running backward and forward passes. The backward-pass updates locally the\n"
      "quadratic approximation of the problem and computes descent direction,\n"
      "and the forward-pass rollouts this new policy by integrating the system dynamics\n"
      "along a tuple of optimized control commands U*.\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<boost::shared_ptr<ShootingProblem> >(bp::args("self", "problem"),
                                                    "Initialize the vector dimension.\n\n"
                                                    ":param problem: shooting problem."))
      .def("solve", &SolverFDDP::solve,
           SolverFDDP_solves(
               bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible", "regInit"),
               "Compute the optimal trajectory xopt, uopt as lists of T+1 and T terms.\n\n"
               "From an initial guess init_xs,init_us (feasible or not), iterate\n"
               "over computeDirection and tryStep until stoppingCriteria is below\n"
               "threshold. It also describes the globalization strategy used\n"
               "during the numerical optimization.\n"
               ":param init_xs: initial guess for state trajectory with T+1 elements (default [])\n"
               ":param init_us: initial guess for control trajectory with T elements (default []).\n"
               ":param maxiter: maximum allowed number of iterations (default 100).\n"
               ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout) (default "
               "False).\n"
               ":param regInit: initial guess for the regularization value. Very low values are typical\n"
               "                used with very good guess points (init_xs, init_us) (default None).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached."))
      .def("updateExpectedImprovement", &SolverFDDP::updateExpectedImprovement,
           bp::return_value_policy<bp::copy_const_reference>(), bp::args("self"),
           "Update the expected improvement model\n\n")
      .add_property("th_acceptNegStep", bp::make_function(&SolverFDDP::get_th_acceptnegstep),
                    bp::make_function(&SolverFDDP::set_th_acceptnegstep),
                    "threshold for step acceptance in ascent direction");
}

}  // namespace python
}  // namespace crocoddyl
