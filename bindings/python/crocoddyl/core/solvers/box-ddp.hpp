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

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverBoxDDP_solves, SolverBoxDDP::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverBoxDDP_computeDirections, SolverDDP::computeDirection, 0, 1)

void exposeSolverBoxDDP() {
  bp::class_<SolverBoxDDP, bp::bases<SolverDDP> >(
      "SolverBoxDDP",
      "Box-constrained DDP solver.\n\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<ShootingProblem&>(bp::args(" self", " problem"),
                                 "Initialize the vector dimension.\n\n"
                                 ":param problem: shooting problem.")[bp::with_custodian_and_ward<1, 2>()])
      .def("solve", &SolverBoxDDP::solve,
           SolverBoxDDP_solves(
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
      .def("computeDirection", &SolverBoxDDP::computeDirection,
           SolverBoxDDP_computeDirections(
               bp::args(" self", " recalc=True"),
               "Compute the search direction (dx, du) for the current guess (xs, us).\n\n"
               "You must call setCandidate first in order to define the current\n"
               "guess. A current guess defines a state and control trajectory\n"
               "(xs, us) of T+1 and T elements, respectively.\n"
               ":params recalc: true for recalculating the derivatives at current state and control.\n"
               ":returns the search direction dx, du and the dual lambdas as lists of T+1, T and T+1 lengths."));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
