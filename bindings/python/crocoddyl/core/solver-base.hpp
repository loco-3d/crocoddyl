///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
#define PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_

#include "crocoddyl/core/solver-base.hpp"
#include "python/crocoddyl/utils.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class SolverAbstract_wrap : public SolverAbstract, public bp::wrapper<SolverAbstract> {
 public:
  using SolverAbstract::problem_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::xreg_;
  using SolverAbstract::ureg_;
  using SolverAbstract::th_acceptstep_;
  using SolverAbstract::th_stop_;

  SolverAbstract_wrap(ShootingProblem& problem) : SolverAbstract(problem), bp::wrapper<SolverAbstract>() {}
  ~SolverAbstract_wrap() {}

  bool solve_wrap(const bp::list& init_xs, const bp::list& init_us,
                  const unsigned int& maxiter, const bool& is_feasible, const double& reg_init) {
    const std::vector<Eigen::VectorXd>& init_xs_vec = python_list_to_std_vector<Eigen::VectorXd>(init_xs);
    const std::vector<Eigen::VectorXd>& init_us_vec = python_list_to_std_vector<Eigen::VectorXd>(init_us);
    return solve(init_xs_vec, init_us_vec, maxiter, is_feasible, reg_init);
  }

  bool solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
             const unsigned int& maxiter, const bool& is_feasible, const double& reg_init) override {
    return bp::call<bool>(this->get_override("solve").ptr(), init_xs, init_us, maxiter, is_feasible, reg_init);
  }

  void computeDirection(const bool& recalc = true) override {
    return bp::call<void>(this->get_override("computeDirection").ptr(), recalc);
  }

  double tryStep(const double& step_length) override {
    return bp::call<double>(this->get_override("tryStep").ptr(), step_length);
  }

  double stoppingCriteria() override {
    return bp::call<double>(this->get_override("stoppingCriteria").ptr());
  }

  const Eigen::Vector2d& expectedImprovement() override {
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

  void setCandidate_wrap(const bp::list& xs_warm, const bp::list& us_warm, const bool& is_feasible = false) {
    const std::vector<Eigen::VectorXd>& xs_warm_vec = python_list_to_std_vector<Eigen::VectorXd>(xs_warm);
    const std::vector<Eigen::VectorXd>& us_warm_vec = python_list_to_std_vector<Eigen::VectorXd>(us_warm);
    setCandidate(xs_warm_vec, us_warm_vec, is_feasible);
  }

  void setCandidate_wrap1(const bp::list& xs_warm, const bp::list& us_warm) {
    setCandidate_wrap(xs_warm, us_warm);
  }

  void setCandidate_wrap2() {
    std::vector<Eigen::VectorXd> xs_warm;
    std::vector<Eigen::VectorXd> us_warm;
    setCandidate(xs_warm, us_warm);
  }

  void setCandidate_wrap3(const bool& is_feasible) {
    std::vector<Eigen::VectorXd> xs_warm;
    std::vector<Eigen::VectorXd> us_warm;
    setCandidate(xs_warm, us_warm, is_feasible);
  }

  // TODO @Carlos this method doesn't match perfectly the Python API, since we need to run
  // problem_$method instead of problem.$method
  // ShootingProblem& get_problem_wrap() { return boost::ref(problem_); }
  double calc_wrap(const bp::list& xs, const bp::list& us) {
    const std::vector<Eigen::VectorXd>& xs_vec = python_list_to_std_vector<Eigen::VectorXd>(xs);
    const std::vector<Eigen::VectorXd>& us_vec = python_list_to_std_vector<Eigen::VectorXd>(us);
    return problem_.calc(xs_vec, us_vec);
  }

  double calcDiff_wrap(const bp::list& xs, const bp::list& us) {
    const std::vector<Eigen::VectorXd>& xs_vec = python_list_to_std_vector<Eigen::VectorXd>(xs);
    const std::vector<Eigen::VectorXd>& us_vec = python_list_to_std_vector<Eigen::VectorXd>(us);
    return problem_.calcDiff(xs_vec, us_vec);
  }

  bp::list rollout_wrap(const bp::list& us) {
    std::vector<Eigen::VectorXd> xs_vec;
    const std::vector<Eigen::VectorXd>& us_vec = python_list_to_std_vector<Eigen::VectorXd>(us);
    problem_.rollout(us_vec, xs_vec);
    return std_vector_to_python_list(xs_vec);
  }

  bp::list get_problem_runningModels() { return std_vector_to_python_list(problem_.running_models_); }

  ActionModelAbstract* get_problem_terminalModel() { return problem_.get_terminalModel(); }

  bp::list get_problem_runningDatas() { return std_vector_to_python_list(problem_.running_datas_); }

  std::shared_ptr<ActionDataAbstract> get_problem_terminalData() { return problem_.get_terminalData(); }

  long unsigned int get_problem_T() { return problem_.get_T(); }

  const Eigen::VectorXd& get_problem_x0() { return problem_.get_x0(); }

  bp::list get_models_wrap() { return std_vector_to_python_list(get_models()); }

  bp::list get_datas_wrap() { return std_vector_to_python_list(get_datas()); }

  bp::list get_xs_wrap() { return std_vector_to_python_list(get_xs()); }

  bp::list get_us_wrap() { return std_vector_to_python_list(get_us()); }

 private:
  Eigen::Vector2d expected_improvement_;
};


void exposeSolverAbstract() {
  bp::class_<SolverAbstract_wrap, boost::noncopyable>(
      "SolverAbstract",
      R"(Abstract class for optimal control solvers.

        In crocoddyl, a solver resolves an optimal control solver which is formulated in a
        problem abstraction. The main routines are computeDirection and tryStep. The former finds
        a search direction and typically computes the derivatives of each action model. The latter
        rollout the dynamics and cost (i.e. the action) to try the search direction found by
        computeDirection. Both functions used the current guess defined by setCandidate. Finally
        solve function is used to define when the search direction and length are computed in each
        iterate. It also describes the globalization strategy (i.e. regularization) of the
        numerical optimization.)",
      bp::init<ShootingProblem&>(bp::args(" self", " problem"),
                                 R"(Initialize the solver model.

:param problem: shooting problem)"))
      .def("solve", pure_virtual(&SolverAbstract_wrap::solve_wrap),
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
      .def("computeDirection", pure_virtual(&SolverAbstract_wrap::computeDirection),
           bp::args(" self", " recalc=True"),
           R"(Compute the search direction (dx, du) for the current guess (xs, us).

You must call setCandidate first in order to define the current
guess. A current guess defines a state and control trajectory
(xs, us) of T+1 and T elements, respectively.
:params recalc: true for recalculating the derivatives at current state and control.
:returns the search direction dx, du and the dual lambdas as lists of T+1, T and T+1 lengths.)")
      .def("tryStep", pure_virtual(&SolverAbstract_wrap::tryStep),
           bp::args(" self", " stepLength"),
           R"(Try a predefined step length and compute its cost improvement.

It uses the search direction found by computeDirection to try a
determined step length; so you need to run first computeDirection.
Additionally it returns the cost improvement along the predefined
step length.
:param stepLength: step length
:returns the cost improvement.)")
      .def("stoppingCriteria", pure_virtual(&SolverAbstract_wrap::stoppingCriteria),
           bp::args(" self"),
           R"(Return a positive value that quantifies the algorithm termination.

These values typically represents the gradient norm which tell us
that it's been reached the local minima. This function is used to
evaluate the algorithm convergence. The stopping criteria strictly
speaking depends on the search direction (calculated by
computeDirection) but it could also depend on the chosen step
length, tested by tryStep.)")
      .def("expectedImprovement", pure_virtual(&SolverAbstract_wrap::expectedImprovement_wrap),
           bp::args(" self"),
           R"(Return the expected improvement from a given current search direction.

For computing the expected improvement, you need to compute first
the search direction by running computeDirection.)")
      .def("setCandidate", &SolverAbstract_wrap::setCandidate,
           bp::args(" self", " xs=None", " us=None", " isFeasible=False"),
           R"(Set the solver candidate warm-point values (xs, us).

The solver candidates are defined as a state and control trajectory
(xs, us) of T+1 and T elements, respectively. Additionally, we need
to define is (xs,us) pair is feasible, this means that the dynamics
rollout give us produces xs.
:param xs: state trajectory of T+1 elements.
:param us: control trajectory of T elements.
:param isFeasible: true if the xs are obtained from integrating the
us (rollout).)")
//       .def("setCallbacks", &SolverAbstract_wrap::setCallbacks),
//            bp::args(" self"),
//            R"(Set a list of callback functions using for diagnostic.

// Each iteration, the solver calls these set of functions in order to
// allowed user the diagnostic of the solver's mperformance.
// :param callbacks: set of callback functions.)")
      .def("setCandidate", &SolverAbstract_wrap::setCandidate_wrap)
      .def("setCandidate", &SolverAbstract_wrap::setCandidate_wrap1)
      .def("setCandidate", &SolverAbstract_wrap::setCandidate_wrap2)
      .def("setCandidate", &SolverAbstract_wrap::setCandidate_wrap3)
      // .add_property("problem", bp::make_getter(&SolverAbstract_wrap::problem_, bp::return_internal_reference<>()), "shooting problem")
      .def("problem_calc", &SolverAbstract_wrap::calc_wrap, bp::args(" self", " xs", " us"),
           R"(Compute the cost and the next states.

First, it computes the next state and cost for each action model
along a state and control trajectory.
:param xs: time-discrete state trajectory
:param us: time-discrete control sequence
:returns the total cost value)")
      .def("problem_calcDiff", &SolverAbstract_wrap::calcDiff_wrap, bp::args(" self", " xs", " us"),
           R"(Compute the cost-and-dynamics derivatives.

These quantities are computed along a given pair of trajectories xs
(states) and us (controls).
:param xs: time-discrete state trajectory
:param us: time-discrete control sequence)")
      .def("problem_rollout", &SolverAbstract_wrap::rollout_wrap, bp::args(" self", " us"),
           R"(Integrate the dynamics given a control sequence.

Rollout the dynamics give a sequence of control commands
:param us: time-discrete control sequence)")
      .add_property("problem_runningModels", bp::make_function(&SolverAbstract_wrap::get_problem_runningModels), "running models")
      .add_property("problem_terminalModel", bp::make_function(&SolverAbstract_wrap::get_problem_terminalModel, bp::return_internal_reference<>()), "terminal model")
      .add_property("problem_runningDatas", bp::make_function(&SolverAbstract_wrap::get_problem_runningDatas), "running datas")
      .add_property("problem_terminalData", bp::make_function(&SolverAbstract_wrap::get_problem_terminalData), "terminal data")
      .add_property("problem_T", bp::make_function(&SolverAbstract_wrap::get_problem_T), "number of nodes")
      .add_property("problem_initialState", bp::make_function(&SolverAbstract_wrap::get_problem_x0, bp::return_value_policy<bp::return_by_value>()), "initial state")
      .def("models", &SolverAbstract_wrap::get_models_wrap, "models")
      .def("datas", &SolverAbstract_wrap::get_datas_wrap, "datas")
      .add_property("xs", &SolverAbstract_wrap::get_xs_wrap, "state trajectory")
      .add_property("us", &SolverAbstract_wrap::get_us_wrap, "control sequence")
      .def_readwrite("isFeasible", &SolverAbstract_wrap::is_feasible_, "feasible (xs,us)")
      .def_readwrite("x_reg", &SolverAbstract_wrap::xreg_, "state regularization")
      .def_readwrite("u_reg", &SolverAbstract_wrap::ureg_, "control regularization")
      .def_readwrite("th_acceptStep", &SolverAbstract_wrap::th_acceptstep_, "threhold for step acceptance")
      .def_readwrite("th_stop", &SolverAbstract_wrap::th_stop_, "threhold for stopping criteria");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_