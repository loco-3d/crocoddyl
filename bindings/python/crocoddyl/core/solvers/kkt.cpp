///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverKKT_solves, SolverKKT::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverKKT_computeDirections, SolverKKT::computeDirection, 0, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverKKT_trySteps, SolverKKT::tryStep, 0, 1)

void exposeSolverKKT() {
  bp::class_<SolverKKT, bp::bases<SolverAbstract> >(
      "SolverKKT",
      "KKT solver.\n\n"
      "The KKT solver computes a primal and dual optimal by inverting\n"
      "the kkt matrix \n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<boost::shared_ptr<ShootingProblem> >(bp::args("self", "problem"),
                                                    "Initialize the vector dimension.\n\n"
                                                    ":param problem: shooting problem."))
      .def("solve", &SolverKKT::solve,
           SolverKKT_solves(
               bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible", "regInit"),
               "Compute the optimal primal(xopt, uopt) and dual(Vx) terms.\n\n"
               ":param init_xs: initial guess for state trajectory with T+1 elements (default []).\n"
               ":param init_us: initial guess for control trajectory with T elements (default []) (default []).\n"
               ":param maxiter: maximun allowed number of iterations (default 100).\n"
               ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout) (default "
               "False).\n"
               ":param regInit: initial guess for the regularization value. Very low values are typical\n"
               "                used with very good guess points (init_xs, init_us) (default None).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached."))
      .def("computeDirection", &SolverKKT::computeDirection,
           SolverKKT_computeDirections(
               bp::args("self", "recalc"),
               "Compute the search direction (dx, du), lambdas for the current guess (xs, us).\n\n"
               "You must call setCandidate first in order to define the current\n"
               "guess. A current guess defines a state and control trajectory\n"
               "(xs, us) of T+1 and T elements, respectively.\n"
               ":params recalc: true for recalculating the derivatives at current state and control.\n"
               ":returns the search direction dx, du and the dual lambdas as lists of T+1, T and T+1 lengths."))
      .def("tryStep", &SolverKKT::tryStep,
           SolverKKT_trySteps(bp::args("self", " stepLength=1"),
                              "Rollout the system with a predefined step length.\n\n"
                              ":param stepLength: step length\n"
                              ":returns the cost improvement."))
      .def("stoppingCriteria", &SolverKKT::stoppingCriteria, bp::args("self"),
           "Return a sum of positive parameters whose sum quantifies the DDP termination.")
      .def("expectedImprovement", &SolverKKT::expectedImprovement, bp::return_value_policy<bp::copy_const_reference>(),
           bp::args("self"),
           "Return two scalars denoting the quadratic improvement model\n\n"
           "For computing the expected improvement, you need to compute first\n"
           "the search direction by running computeDirection. The quadratic\n"
           "improvement model is described as dV = f_0 - f_+ = d1*a + d2*a**2/2.")
      .add_property("kkt", make_function(&SolverKKT::get_kkt, bp::return_value_policy<bp::copy_const_reference>()),
                    "kkt")
      .add_property("kktref",
                    make_function(&SolverKKT::get_kktref, bp::return_value_policy<bp::copy_const_reference>()),
                    "kktref")
      .add_property("primaldual",
                    make_function(&SolverKKT::get_primaldual, bp::return_value_policy<bp::copy_const_reference>()),
                    "primaldual")
      .add_property("lambdas",
                    make_function(&SolverKKT::get_lambdas, bp::return_value_policy<bp::copy_const_reference>()),
                    "lambdas")
      .add_property("dxs", make_function(&SolverKKT::get_dxs, bp::return_value_policy<bp::copy_const_reference>()),
                    "dxs")
      .add_property("dus", make_function(&SolverKKT::get_dus, bp::return_value_policy<bp::copy_const_reference>()),
                    "dus");
}

}  // namespace python
}  // namespace crocoddyl
