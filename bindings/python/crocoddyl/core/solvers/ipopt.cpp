#ifdef CROCODDYL_WITH_IPOPT

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/solvers/ipopt.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverIpopt_solves, SolverIpopt::solve, 0, 5)

void exposeSolverIpopt() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverIpopt>>();
  bp::class_<SolverIpopt, bp::bases<SolverAbstract>>(
      "SolverIpopt",
      bp::init<const boost::shared_ptr<crocoddyl::ShootingProblem>&>(bp::args("self", "problem"), "Initialize solver"))
      .def("solve", &SolverIpopt::solve,
           SolverIpopt_solves(
               bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible", "regInit"),
               "Compute the optimal trajectory xopt, uopt as lists of T+1 and T terms.\n\n"
               "From an initial guess init_xs,init_us (feasible or not), iterate\n"
               "over computeDirection and tryStep until stoppingCriteria is below\n"
               "threshold. It also describes the globalization strategy used\n"
               "during the numerical optimization.\n"
               ":param init_xs: initial guess for state trajectory with T+1 elements (default []).\n"
               ":param init_us: initial guess for control trajectory with T elements (default []) (default []).\n"
               ":param maxiter: maximum allowed number of iterations (default 100).\n"
               ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout) (default "
               "False).\n"
               ":param regInit: initial guess for the regularization value. Very low values are typical\n"
               "                used with very good guess points (init_xs, init_us) (default None).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached."));
}

}  // namespace python
}  // namespace crocoddyl

#endif