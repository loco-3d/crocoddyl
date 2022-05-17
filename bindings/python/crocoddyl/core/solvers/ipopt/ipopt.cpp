#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/solvers/ipopt/ipopt.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverIpOpt_solves, SolverIpOpt::solve, 0, 5)

void exposeSolverIpOpt() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverIpOpt>>();
  bp::class_<SolverIpOpt, bp::bases<SolverAbstract>>(
      "SolverIpOpt",
      bp::init<const boost::shared_ptr<crocoddyl::ShootingProblem>&>(bp::args("self", "problem"), "Initialize solver"))
      .def("solve", &SolverIpOpt::solve,
           SolverIpOpt_solves(
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
