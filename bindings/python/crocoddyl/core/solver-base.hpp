
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_

#include "crocoddyl/core/solver-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class SolverAbstractTpl_wrap : public SolverAbstractTpl<_Scalar>,
                               public bp::wrapper<SolverAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector2s Vector2s;

  using SolverAbstract::cost_;
  using SolverAbstract::cost_try_;
  using SolverAbstract::d_;
  using SolverAbstract::dfeas_;
  using SolverAbstract::dPhi_;
  using SolverAbstract::dPhiexp_;
  using SolverAbstract::DV_;
  using SolverAbstract::dV_;
  using SolverAbstract::dVexp_;
  using SolverAbstract::dVexp_full_;
  using SolverAbstract::feas_;
  using SolverAbstract::ffeas_;
  using SolverAbstract::ffeas_try_;
  using SolverAbstract::fs_;
  using SolverAbstract::fs_try_;
  using SolverAbstract::gfeas_;
  using SolverAbstract::gfeas_try_;
  using SolverAbstract::hfeas_;
  using SolverAbstract::hfeas_try_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::merit_;
  using SolverAbstract::problem_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::us_;
  using SolverAbstract::us_try_;
  using SolverAbstract::xs_;
  using SolverAbstract::xs_try_;

  explicit SolverAbstractTpl_wrap(std::shared_ptr<ShootingProblem> problem)
      : SolverAbstract(problem), bp::wrapper<SolverAbstract>() {}
  ~SolverAbstractTpl_wrap() = default;

  bool solve(
      const std::vector<VectorXs>& init_xs = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& init_us = DefaultVector<Scalar>::value,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const Scalar reg_init =
          std::numeric_limits<Scalar>::quiet_NaN()) override {
    if (bp::override solve = this->get_override("solve")) {
      return bp::call<bool>(solve.ptr(), init_xs, init_us, maxiter, is_feasible,
                            reg_init);
    }
    return this->SolverAbstract::solve(init_xs, init_us, maxiter, is_feasible,
                                       reg_init);
  }

  void computeDirection(const bool recalc = true) override {
    return bp::call<void>(this->get_override("computeDirection").ptr(), recalc);
  }

  Scalar tryStep(const Scalar step_length = Scalar(1.)) override {
    return bp::call<Scalar>(this->get_override("tryStep").ptr(), step_length);
  }

  Scalar stoppingCriteria() override {
    stop_ = bp::call<Scalar>(this->get_override("stoppingCriteria").ptr());
    return stop_;
  }

  void increaseRegularization() override {
    bp::call<void>(this->get_override("increaseRegularization").ptr());
  }

  void decreaseRegularization() override {
    bp::call<void>(this->get_override("decreaseRegularization").ptr());
  }

  void updateCandidate() override {
    bp::call<void>(this->get_override("updateCandidate").ptr());
  }

  bool checkAcceptance() override {
    return bp::call<bool>(this->get_override("checkAcceptance").ptr());
  }

  void updateMeritFunction() override {
    bp::call<void>(this->get_override("updateMeritFunction").ptr());
  }

  const Vector2s& expectedImprovement() override {
    d_ << bp::call<Eigen::Matrix<Scalar, 2, 1>>(
        this->get_override("expectedImprovement").ptr());
    return d_;
  }

  bool default_solve(
      const std::vector<VectorXs>& init_xs = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& init_us = DefaultVector<Scalar>::value,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const Scalar reg_init = std::numeric_limits<Scalar>::quiet_NaN()) {
    return this->SolverAbstract::solve(init_xs, init_us, maxiter, is_feasible,
                                       reg_init);
  }

  void allocateData() { SolverAbstract::allocateData(); }

  template <typename NewScalar>
  SolverAbstractTpl_wrap<NewScalar> cast() const {
    typedef SolverAbstractTpl_wrap<NewScalar> ReturnType;
    typedef ShootingProblemTpl<NewScalar> ProblemType;
    ReturnType ret(
        std::make_shared<ProblemType>(problem_->template cast<NewScalar>()));
    return ret;
  }
};

template <typename _Scalar>
class CallbackAbstractTpl_wrap
    : public CallbackAbstractTpl<_Scalar>,
      public bp::wrapper<CallbackAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(CallbackBase, CallbackAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;

  CallbackAbstractTpl_wrap()
      : CallbackAbstract(), bp::wrapper<CallbackAbstract>() {}
  ~CallbackAbstractTpl_wrap() = default;

  void operator()(SolverAbstract& solver) override {
    return bp::call<void>(this->get_override("__call__").ptr(),
                          boost::ref(solver));
  }

  template <typename NewScalar>
  CallbackAbstractTpl_wrap<NewScalar> cast() const {
    typedef CallbackAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret;
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVER_BASE_HPP_
