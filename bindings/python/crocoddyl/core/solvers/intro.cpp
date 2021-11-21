///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/solvers/intro.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverIntro_solves, SolverIntro::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverIntro_trySteps, SolverIntro::tryStep, 0, 1)

void exposeSolverIntro() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverIntro> >();

  bp::enum_<EqualitySolverType>("EqualitySolverType")
      .value("LuNull", LuNull)
      .value("QrNull", QrNull)
      .value("Schur", Schur)
      .export_values();

  bp::class_<SolverIntro, bp::bases<SolverDDP> >(
      "SolverIntro", bp::init<boost::shared_ptr<ShootingProblem> >(bp::args("self", "problem"),
                                                                   "Initialize the vector dimension.\n\n"
                                                                   ":param problem: shooting problem."))
      .def("solve", &SolverIntro::solve,
           SolverIntro_solves(
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
      .def("tryStep", &SolverIntro::tryStep,
           SolverIntro_trySteps(bp::args("self", "stepLength"),
                                "Rollout the system with a predefined step length.\n\n"
                                ":param stepLength: step length (default 1)\n"
                                ":returns the cost improvement."))
      .add_property("eq_solver", bp::make_function(&SolverIntro::get_equality_solver),
                    bp::make_function(&SolverIntro::set_equality_solver),
                    "type of solver used for handling the equality constraints")
      .add_property("rho", bp::make_function(&SolverIntro::get_rho), bp::make_function(&SolverIntro::set_rho),
                    "parameter used in the merit function to predict the expected reduction")
      .add_property("dPhi", bp::make_function(&SolverIntro::get_dPhi), "reduction in the merit function")
      .add_property("dPhiexp", bp::make_function(&SolverIntro::get_dPhiexp),
                    "expected reduction in the merit function")
      .add_property("upsilon", bp::make_function(&SolverIntro::get_upsilon),
                    "estimated penalty paramter that balances relative contribution of the cost function and equality "
                    "constraints");
}

}  // namespace python
}  // namespace crocoddyl
