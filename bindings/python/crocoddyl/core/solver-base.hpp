///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_

#include <vector>
#include <memory>
#include "crocoddyl/core/solver-base.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class SolverAbstract_wrap : public SolverAbstract, public bp::wrapper<SolverAbstract> {
 public:
  using SolverAbstract::cost_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::steplength_;

  explicit SolverAbstract_wrap(boost::shared_ptr<ShootingProblem> problem)
      : SolverAbstract(problem), bp::wrapper<SolverAbstract>() {}
  ~SolverAbstract_wrap() {}

  bool solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
             const std::size_t& maxiter, const bool& is_feasible, const double& reg_init) {
    return bp::call<bool>(this->get_override("solve").ptr(), init_xs, init_us, maxiter, is_feasible, reg_init);
  }

  void computeDirection(const bool& recalc = true) {
    return bp::call<void>(this->get_override("computeDirection").ptr(), recalc);
  }

  double tryStep(const double& step_length) {
    return bp::call<double>(this->get_override("tryStep").ptr(), step_length);
  }

  double stoppingCriteria() { return bp::call<double>(this->get_override("stoppingCriteria").ptr()); }

  const Eigen::Vector2d& expectedImprovement() {
    bp::list exp_impr = bp::call<bp::list>(this->get_override("expectedImprovement").ptr());
    expected_improvement_ << bp::extract<double>(exp_impr[0]), bp::extract<double>(exp_impr[1]);
    return expected_improvement_;
  }

  bp::list expectedImprovement_wrap() {
    expectedImprovement();
    bp::list exp_impr;
    exp_impr.append(expected_improvement_[0]);
    exp_impr.append(expected_improvement_[1]);
    return exp_impr;
  }

 private:
  Eigen::Vector2d expected_improvement_;
};

class CallbackAbstract_wrap : public CallbackAbstract, public bp::wrapper<CallbackAbstract> {
 public:
  CallbackAbstract_wrap() : CallbackAbstract(), bp::wrapper<CallbackAbstract>() {}
  ~CallbackAbstract_wrap() {}

  void operator()(SolverAbstract& solver) {
    return bp::call<void>(this->get_override("__call__").ptr(), boost::ref(solver));
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setCandidate_overloads, SolverAbstract::setCandidate, 0, 3)

void exposeSolverAbstract() {
  // Register custom converters between std::vector and Python list
  typedef boost::shared_ptr<CallbackAbstract> CallbackAbstractPtr;
  bp::to_python_converter<std::vector<CallbackAbstractPtr, std::allocator<CallbackAbstractPtr> >,
                          vector_to_list<CallbackAbstractPtr, false> >();
  list_to_vector().from_python<std::vector<CallbackAbstractPtr, std::allocator<CallbackAbstractPtr> > >();

  bp::class_<SolverAbstract_wrap, boost::noncopyable>(
      "SolverAbstract",
      "Abstract class for optimal control solvers.\n\n"
      "In crocoddyl, a solver resolves an optimal control solver which is formulated in a\n"
      "problem abstraction. The main routines are computeDirection and tryStep. The former finds\n"
      "a search direction and typically computes the derivatives of each action model. The latter\n"
      "rollout the dynamics and cost (i.e. the action) to try the search direction found by\n"
      "computeDirection. Both functions used the current guess defined by setCandidate. Finally\n"
      "solve function is used to define when the search direction and length are computed in each\n"
      "iterate. It also describes the globalization strategy (i.e. regularization) of the\n"
      "numerical optimization.",
      bp::init<boost::shared_ptr<ShootingProblem> >(bp::args("self", "problem"),
                                                    "Initialize the solver model.\n\n"
                                                    ":param problem: shooting problem"))
      .def("solve", pure_virtual(&SolverAbstract_wrap::solve),
           bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible", "regInit"),
           "Compute the optimal trajectory xopt,uopt as lists of T+1 and T terms.\n\n"
           "From an initial guess init_xs,init_us (feasible or not), iterate\n"
           "over computeDirection and tryStep until stoppingCriteria is below\n"
           "threshold. It also describes the globalization strategy used\n"
           "during the numerical optimization.\n"
           ":param init_xs: initial guess for state trajectory with T+1 elements (default [])\n"
           ":param init_us: initial guess for control trajectory with T elements (default []).\n"
           ":param maxiter: maximun allowed number of iterations (default 100).\n"
           ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout) (default "
           "False).\n"
           ":param regInit: initial guess for the regularization value. Very low\n"
           "                values are typical used with very good guess points (init_xs, init_us).\n"
           ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached.")
      .def("computeDirection", pure_virtual(&SolverAbstract_wrap::computeDirection), bp::args("self", "recalc"),
           "Compute the search direction (dx, du) for the current guess (xs, us).\n\n"
           "You must call setCandidate first in order to define the current\n"
           "guess. A current guess defines a state and control trajectory\n"
           "(xs, us) of T+1 and T elements, respectively.\n"
           ":params recalc: true for recalculating the derivatives at current state and control.\n"
           ":returns the search direction dx, du and the dual lambdas as lists of T+1, T and T+1 lengths.")
      .def("tryStep", pure_virtual(&SolverAbstract_wrap::tryStep), bp::args("self", " stepLength"),
           "Try a predefined step length and compute its cost improvement.\n\n"
           "It uses the search direction found by computeDirection to try a\n"
           "determined step length; so you need to run first computeDirection.\n"
           "Additionally it returns the cost improvement along the predefined\n"
           "step length.\n"
           ":param stepLength: step length\n"
           ":returns the cost improvement.")
      .def("stoppingCriteria", pure_virtual(&SolverAbstract_wrap::stoppingCriteria), bp::args("self"),
           "Return a positive value that quantifies the algorithm termination.\n\n"
           "These values typically represents the gradient norm which tell us\n"
           "that it's been reached the local minima. This function is used to\n"
           "evaluate the algorithm convergence. The stopping criteria strictly\n"
           "speaking depends on the search direction (calculated by\n"
           "computeDirection) but it could also depend on the chosen step\n"
           "length, tested by tryStep.")
      .def("expectedImprovement", pure_virtual(&SolverAbstract_wrap::expectedImprovement_wrap), bp::args("self"),
           "Return the expected improvement from a given current search direction.\n\n"
           "For computing the expected improvement, you need to compute first\n"
           "the search direction by running computeDirection.")
      .def("setCandidate", &SolverAbstract_wrap::setCandidate,
           setCandidate_overloads(bp::args("self", "xs", "us", "isFeasible"),
                                  "Set the solver candidate warm-point values (xs, us).\n\n"
                                  "The solver candidates are defined as a state and control trajectory\n"
                                  "(xs, us) of T+1 and T elements, respectively. Additionally, we need\n"
                                  "to define is (xs,us) pair is feasible, this means that the dynamics\n"
                                  "rollout give us produces xs.\n"
                                  ":param xs: state trajectory of T+1 elements (default []).\n"
                                  ":param us: control trajectory of T elements (default []).\n"
                                  ":param isFeasible: true if the xs are obtained from integrating the\n"
                                  "us (rollout)."))
      .def("setCallbacks", &SolverAbstract_wrap::setCallbacks, bp::args("self"),
           "Set a list of callback functions using for diagnostic.\n\n"
           "Each iteration, the solver calls these set of functions in order to\n"
           "allowed user the diagnostic of the solver's performance.\n"
           ":param callbacks: set of callback functions.")
      .def("getCallbacks", &SolverAbstract_wrap::getCallbacks, bp::return_value_policy<bp::copy_const_reference>(),
           bp::args("self"),
           "Return the list of callback functions using for diagnostic.\n\n"
           ":return set of callback functions.")
      .add_property(
          "problem",
          bp::make_function(&SolverAbstract_wrap::get_problem, bp::return_value_policy<bp::copy_const_reference>()),
          "shooting problem")
      .def("models", &SolverAbstract_wrap::get_models, bp::return_value_policy<bp::copy_const_reference>(), "models")
      .def("datas", &SolverAbstract_wrap::get_datas, bp::return_value_policy<bp::copy_const_reference>(), "datas")
      .add_property(
          "xs", bp::make_function(&SolverAbstract_wrap::get_xs, bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverAbstract_wrap::set_xs), "state trajectory")
      .add_property(
          "us", bp::make_function(&SolverAbstract_wrap::get_us, bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverAbstract_wrap::set_us), "control sequence")
      .def_readwrite("isFeasible", &SolverAbstract_wrap::is_feasible_, "feasible (xs,us)")
      .def_readwrite("cost", &SolverAbstract_wrap::cost_, "total cost")
      .add_property(
          "x_reg",
          bp::make_function(&SolverAbstract_wrap::get_xreg, bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverAbstract_wrap::set_xreg), "state regularization")
      .add_property(
          "u_reg",
          bp::make_function(&SolverAbstract_wrap::get_ureg, bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverAbstract_wrap::set_ureg), "control regularization")
      .def_readwrite("stepLength", &SolverAbstract_wrap::steplength_, "applied step length")      .add_property("th_acceptStep",
                    bp::make_function(&SolverAbstract_wrap::get_th_acceptstep,
                                      bp::return_value_policy<bp::copy_const_reference>()),
                    bp::make_function(&SolverAbstract_wrap::set_th_acceptstep), "threshold for step acceptance")
      .add_property(
          "th_stop",
          bp::make_function(&SolverAbstract_wrap::get_th_stop, bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverAbstract_wrap::set_th_stop), "threshold for stopping criteria")
      .def_readwrite("iter", &SolverAbstract_wrap::iter_, "number of iterations runned in solve()");

  bp::class_<CallbackAbstract_wrap, boost::noncopyable>(
      "CallbackAbstract",
      "Abstract class for solver callbacks.\n\n"
      "A callback is used to diagnostic the behaviour of our solver in each iteration of it.\n"
      "For instance, it can be used to print values, record data or display motions")
      .def("__call__", pure_virtual(&CallbackAbstract_wrap::operator()), bp::args("self", "solver"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
