///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_SOLVERS_DDP_HPP_
#define PYTHON_CROCODDYL_CORE_SOLVERS_DDP_HPP_

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverDDP_computeDirections, SolverDDP::computeDirection, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverDDP_trySteps, SolverDDP::tryStep, 0, 1)

void exposeSolverDDP() {
  bp::class_<SolverDDP, bp::bases<SolverAbstract>>(
      "SolverDDP",
      R"(DDP solver.

        The DDP solver computes an optimal trajectory and control commands by iteratives
        running backward and forward passes. The backward-pass updates locally the
        quadratic approximation of the problem and computes descent direction,
        and the forward-pass rollouts this new policy by integrating the system dynamics
        along a tuple of optimized control commands U*.
        :param shootingProblem: shooting problem (list of action models along trajectory).)",
      bp::init<ShootingProblem&>(bp::args(" self", " problem"),
                                 R"(Initialize the vector dimension.

:param problem: shooting problem.)"))
      .def("solve", &SolverDDP::solve,
           bp::args(" self", " init_xs=None", " init_us=None", " maxiter=100", " isFeasible=False", " regInit=None"),
           R"(Compute the optimal trajectory xopt,uopt as lists of T+1 and T terms.

From an initial guess init_xs,init_us (feasible or not), iterate
over computeDirection and tryStep until stoppingCriteria is below
threshold. It also describes the globalization strategy used
during the numerical optimization.
:param init_xs: initial guess for state trajectory with T+1 elements.
:param init_us: initial guess for control trajectory with T elements.
:param maxiter: maximun allowed number of iterations.
:param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout).
:param regInit: initial guess for the regularization value. Very low values are typical used with very good
guess points (init_xs, init_us).
:returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached.)")
      .def("computeDirection", &SolverDDP::computeDirection, SolverDDP_computeDirections(
           bp::args(" self", " recalc=True"),
           R"(Compute the search direction (dx, du) for the current guess (xs, us).

You must call setCandidate first in order to define the current
guess. A current guess defines a state and control trajectory
(xs, us) of T+1 and T elements, respectively.
:params recalc: true for recalculating the derivatives at current state and control.
:returns the search direction dx, du and the dual lambdas as lists of T+1, T and T+1 lengths.)"))
      .def("tryStep", &SolverDDP::tryStep, SolverDDP_trySteps(
           bp::args(" self", " stepLength=1"),
           R"(Rollout the system with a predefined step length.

:param stepLength: step length
:returns the cost improvement.)"))
      .def("stoppingCriteria", &SolverDDP::stoppingCriteria,
           bp::args(" self"),
           R"(Return a sum of positive parameters whose sum quantifies the DDP termination.)")
      .def("expectedImprovement", &SolverDDP::expectedImprovement, bp::return_value_policy<bp::copy_const_reference>(),
           bp::args(" self"),
           R"(Return two scalars denoting the quadratic improvement model

For computing the expected improvement, you need to compute first
the search direction by running computeDirection. The quadratic 
improvement model is described as dV = f_0 - f_+ = d1*a + d2*a**2/2.)")
      .add_property("Vxx", make_function(&SolverDDP::get_Vxx, bp::return_value_policy<bp::copy_const_reference>()), "Vxx")
      .add_property("Vx", make_function(&SolverDDP::get_Vx, bp::return_value_policy<bp::copy_const_reference>()), "Vx")
      .add_property("Qxx", make_function(&SolverDDP::get_Qxx, bp::return_value_policy<bp::copy_const_reference>()), "Qxx")
      .add_property("Qxu", make_function(&SolverDDP::get_Qxu, bp::return_value_policy<bp::copy_const_reference>()), "Qxu")
      .add_property("Quu", make_function(&SolverDDP::get_Quu, bp::return_value_policy<bp::copy_const_reference>()), "Quu")
      .add_property("Qx", make_function(&SolverDDP::get_Qx, bp::return_value_policy<bp::copy_const_reference>()), "Qx")
      .add_property("Qu", make_function(&SolverDDP::get_Qu, bp::return_value_policy<bp::copy_const_reference>()), "Qu")
      .add_property("K", make_function(&SolverDDP::get_K, bp::return_value_policy<bp::copy_const_reference>()), "K")
      .add_property("k", make_function(&SolverDDP::get_k, bp::return_value_policy<bp::copy_const_reference>()), "k");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_SOLVERS_DDP_HPP_