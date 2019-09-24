///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_FDDP_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_FDDP_HPP_

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverFDDP_solves, SolverFDDP::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverFDDP_trySteps, SolverFDDP::tryStep, 0, 1)

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
      .def("solve", &SolverFDDP::solve,
           SolverFDDP_solves(
               bp::args(" self", " init_xs=[]", " init_us=[]", " maxiter=100", " isFeasible=False", " regInit=None"),
               "Compute the optimal trajectory xopt, uopt as lists of T+1 and T terms.\n\n"
               "From an initial guess init_xs,init_us (feasible or not), iterate\n"
               "over computeDirection and tryStep until stoppingCriteria is below\n"
               "threshold. It also describes the globalization strategy used\n"
               "during the numerical optimization.\n"
               ":param init_xs: initial guess for state trajectory with T+1 elements.\n"
               ":param init_us: initial guess for control trajectory with T elements.\n"
               ":param maxiter: maximun allowed number of iterations.\n"
               ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout).\n"
               ":param regInit: initial guess for the regularization value. Very low values are typical\n"
               "                used with very good guess points (init_xs, init_us).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached."))
      .def("tryStep", &SolverFDDP::tryStep,
           SolverFDDP_trySteps(bp::args(" self", " stepLength=1"),
                               "Rollout the system with a predefined step length.\n\n"
                               ":param stepLength: step length\n"
                               ":returns the cost improvement."))
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
      .def("calc", &SolverFDDP::calc, bp::args(" self"),
           "Update the Jacobian and Hessian of the optimal control problem\n\n"
           "These derivatives are computed around the guess state and control\n"
           "trajectory. These trajectory can be set by using setCandidate.\n"
           ":return the total cost around the guess trajectory.")
      .def("forwardPass", &SolverFDDP::forwardPass, bp::args(" self", " stepLength=1"),
           "Run the forward pass or rollout\n\n"
           "It rollouts the action model give the computed policy (feedfoward terns and feedback\n"
           "gains) by the backwardPass. We can define different step lengths\n"
           ":param stepLength: applied step length (<= 1. and >= 0.)");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_FDDP_HPP_
