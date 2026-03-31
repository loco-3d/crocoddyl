
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
  typedef typename MathBase::Vector3s Vector3s;

  using SolverAbstract::acceptstep_;
  using SolverAbstract::cost_;
  using SolverAbstract::cost_try_;
  using SolverAbstract::dfeas_;
  using SolverAbstract::dImpr_;
  using SolverAbstract::dPhi_;
  using SolverAbstract::dPhiexp_;
  using SolverAbstract::dus_;
  using SolverAbstract::DV_;
  using SolverAbstract::dV_;
  using SolverAbstract::dVexp_;
  using SolverAbstract::dVexp_full_;
  using SolverAbstract::dxs_;
  using SolverAbstract::feas_;
  using SolverAbstract::ffeas_;
  using SolverAbstract::ffeas_try_;
  using SolverAbstract::fs_;
  using SolverAbstract::fs_try_;
  using SolverAbstract::g_adj_;
  using SolverAbstract::gfeas_;
  using SolverAbstract::gfeas_try_;
  using SolverAbstract::hfeas_;
  using SolverAbstract::hfeas_try_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::merit_;
  using SolverAbstract::ng_T_;
  using SolverAbstract::nh_T_;
  using SolverAbstract::problem_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::u_adj_;
  using SolverAbstract::us_;
  using SolverAbstract::us_try_;
  using SolverAbstract::x_adj_;
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
    if (bp::override computeDirection =
            this->get_override("computeDirection")) {
      return bp::call<void>(computeDirection.ptr(), recalc);
    } else {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "SolverAbstract.computeDirection() must be implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  Scalar tryStep(const Scalar step_length = Scalar(1.)) override {
    if (bp::override tryStep = this->get_override("tryStep")) {
      return bp::call<Scalar>(tryStep.ptr(), step_length);
    } else {
      return this->SolverAbstract::tryStep(step_length);
    }
  }

  void computeCandidate(const Scalar step_length = Scalar(1.)) override {
    if (bp::override computeCandidate =
            this->get_override("computeCandidate")) {
      return bp::call<void>(this->get_override("computeCandidate").ptr(),
                            step_length);
    } else {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "SolverAbstract.computeCandidate() must be implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  Scalar stoppingCriteria() override {
    if (bp::override stoppingCriteria =
            this->get_override("stoppingCriteria")) {
      stop_ = bp::call<Scalar>(stoppingCriteria.ptr());
    } else {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "SolverAbstract.stoppingCriteria() must be implemented in subclass.");
      bp::throw_error_already_set();
    }
    return stop_;
  }

  Vector3s expectedImprovement() override {
    if (bp::override expectedImprovement =
            this->get_override("expectedImprovement")) {
      DV_ = bp::call<Vector3s>(expectedImprovement.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.expectedImprovement() must be "
                      "implemented in subclass.");
      bp::throw_error_already_set();
    }
    return DV_;
  }

  void computeMeritFunctionImprovement() override {
    if (bp::override computeMeritFunctionImprovement =
            this->get_override("computeMeritFunctionImprovement")) {
      return bp::call<void>(computeMeritFunctionImprovement.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.computeMeritFunctionImprovement() must "
                      "be implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  void computeExpectedMeritFunctionImprovement() override {
    if (bp::override computeExpectedMeritFunctionImprovement =
            this->get_override("computeExpectedMeritFunctionImprovement")) {
      return bp::call<void>(computeExpectedMeritFunctionImprovement.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.computeExpectedMeritFunctionImprovement()"
                      " must be implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  void updateMeritFunction() override {
    if (bp::override updateMeritFunction =
            this->get_override("updateMeritFunction")) {
      return bp::call<void>(updateMeritFunction.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.updateMeritFunction() must be "
                      "implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  bool checkAcceptance() override {
    if (bp::override checkAcceptance = this->get_override("checkAcceptance")) {
      return bp::call<bool>(checkAcceptance.ptr());
    } else {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "SolverAbstract.checkAcceptance() must be implemented in subclass.");
      bp::throw_error_already_set();
    }
    return false;
  }

  void calcDir() override {
    if (bp::override calcDir = this->get_override("calcDir")) {
      return bp::call<void>(calcDir.ptr());
    } else {
      return this->SolverAbstract::calcDir();
    }
  }

  void setCandidate(
      const std::vector<VectorXs>& xs_warm = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& us_warm = DefaultVector<Scalar>::value,
      const bool is_feasible = false) override {
    if (bp::override setCandidate = this->get_override("setCandidate")) {
      return bp::call<void>(setCandidate.ptr(), xs_warm, us_warm, is_feasible);
    } else {
      return this->SolverAbstract::setCandidate(xs_warm, us_warm, is_feasible);
    }
  }

  void updateCandidate() override {
    if (bp::override updateCandidate = this->get_override("updateCandidate")) {
      return bp::call<void>(updateCandidate.ptr());
    } else {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "SolverAbstract.updateCandidate() must be implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  bool increaseRegularizationCriteria() override {
    if (bp::override increaseRegularizationCriteria =
            this->get_override("increaseRegularizationCriteria")) {
      return bp::call<bool>(increaseRegularizationCriteria.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.increaseRegularizationCriteria() must be "
                      "implemented in subclass.");
      bp::throw_error_already_set();
    }
    return false;
  }

  bool decreaseRegularizationCriteria() override {
    if (bp::override decreaseRegularizationCriteria =
            this->get_override("decreaseRegularizationCriteria")) {
      return bp::call<bool>(decreaseRegularizationCriteria.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.decreaseRegularizationCriteria() must be "
                      "implemented in subclass.");
      bp::throw_error_already_set();
    }
    return false;
  }

  void increaseRegularization() override {
    if (bp::override increaseRegularization =
            this->get_override("increaseRegularization")) {
      return bp::call<void>(increaseRegularization.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.increaseRegularization() must be "
                      "implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  void decreaseRegularization() override {
    if (bp::override decreaseRegularization =
            this->get_override("decreaseRegularization")) {
      return bp::call<void>(decreaseRegularization.ptr());
    } else {
      PyErr_SetString(PyExc_NotImplementedError,
                      "SolverAbstract.decreaseRegularization() must be "
                      "implemented in subclass.");
      bp::throw_error_already_set();
    }
  }

  void resizeRunningData() override {
    if (bp::override resizeRunningData =
            this->get_override("resizeRunningData")) {
      return bp::call<void>(resizeRunningData.ptr());
    } else {
      return this->SolverAbstract::resizeRunningData();
    }
  }

  void resizeTerminalData() override {
    if (bp::override resizeTerminalData =
            this->get_override("resizeTerminalData")) {
      return bp::call<void>(resizeTerminalData.ptr());
    } else {
      return this->SolverAbstract::resizeTerminalData();
    }
  }

  bool default_solve(
      const std::vector<VectorXs>& init_xs = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& init_us = DefaultVector<Scalar>::value,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const Scalar reg_init = std::numeric_limits<Scalar>::quiet_NaN()) {
    return this->SolverAbstract::solve(init_xs, init_us, maxiter, is_feasible,
                                       reg_init);
  }

  Scalar default_tryStep(const Scalar step_length = Scalar(1.)) {
    return this->SolverAbstract::tryStep(step_length);
  }

  void default_calcDir() { return this->SolverAbstract::calcDir(); }

  void default_setCandidate(
      const std::vector<VectorXs>& xs_warm = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& us_warm = DefaultVector<Scalar>::value,
      const bool is_feasible = false) {
    return this->SolverAbstract::setCandidate(xs_warm, us_warm, is_feasible);
  }

  void default_resizeRunningData() {
    return this->SolverAbstract::resizeRunningData();
  }

  void default_resizeTerminalData() {
    return this->SolverAbstract::resizeTerminalData();
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
