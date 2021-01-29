
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_

#include "crocoddyl/core/solver-base.hpp"
#include <memory>
#include <vector>

namespace crocoddyl {
namespace python {

class SolverAbstract_wrap : public SolverAbstract,
                            public bp::wrapper<SolverAbstract> {
public:
  using SolverAbstract::cost_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::steplength_;

  explicit SolverAbstract_wrap(boost::shared_ptr<ShootingProblem> problem)
      : SolverAbstract(problem), bp::wrapper<SolverAbstract>() {}
  ~SolverAbstract_wrap() {}

  bool solve(const std::vector<Eigen::VectorXd> &init_xs,
             const std::vector<Eigen::VectorXd> &init_us,
             const std::size_t &maxiter, const bool &is_feasible,
             const double &reg_init) {
    return bp::call<bool>(this->get_override("solve").ptr(), init_xs, init_us,
                          maxiter, is_feasible, reg_init);
  }

  void computeDirection(const bool &recalc = true) {
    return bp::call<void>(this->get_override("computeDirection").ptr(), recalc);
  }

  double tryStep(const double &step_length = 1) {
    return bp::call<double>(this->get_override("tryStep").ptr(), step_length);
  }

  double stoppingCriteria() {
    return bp::call<double>(this->get_override("stoppingCriteria").ptr());
  }

  const Eigen::Vector2d &expectedImprovement() {
    bp::list exp_impr =
        bp::call<bp::list>(this->get_override("expectedImprovement").ptr());
    expected_improvement_ << bp::extract<double>(exp_impr[0]),
        bp::extract<double>(exp_impr[1]);
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

class CallbackAbstract_wrap : public CallbackAbstract,
                              public bp::wrapper<CallbackAbstract> {
public:
  CallbackAbstract_wrap()
      : CallbackAbstract(), bp::wrapper<CallbackAbstract>() {}
  ~CallbackAbstract_wrap() {}

  void operator()(SolverAbstract &solver) {
    return bp::call<void>(this->get_override("__call__").ptr(),
                          boost::ref(solver));
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setCandidate_overloads,
                                       SolverAbstract::setCandidate, 0, 3)

} // namespace python
} // namespace crocoddyl

#endif // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
