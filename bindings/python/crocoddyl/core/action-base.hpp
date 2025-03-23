///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class ActionModelAbstractTpl_wrap
    : public ActionModelAbstractTpl<_Scalar>,
      public bp::wrapper<ActionModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ActionModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef typename crocoddyl::ActionModelAbstractTpl<Scalar> ActionModel;
  typedef typename crocoddyl::ActionDataAbstractTpl<Scalar> ActionData;
  typedef typename crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename ActionModel::VectorXs VectorXs;
  using ActionModel::ng_;
  using ActionModel::ng_T_;
  using ActionModel::nh_;
  using ActionModel::nh_T_;
  using ActionModel::nr_;
  using ActionModel::nu_;
  using ActionModel::state_;
  using ActionModel::unone_;

  ActionModelAbstractTpl_wrap(std::shared_ptr<State> state,
                              const std::size_t nu, const std::size_t nr = 1,
                              const std::size_t ng = 0,
                              const std::size_t nh = 0,
                              const std::size_t ng_T = 0,
                              const std::size_t nh_T = 0)
      : ActionModel(state, nu, nr, ng, nh, ng_T, nh_T),
        bp::wrapper<ActionModel>() {
    unone_ = VectorXs::Constant(nu, Scalar(NAN));
  }

  void calc(const std::shared_ptr<ActionData>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    if (std::isnan(
            scalar_cast<ScalarType>(u.template lpNorm<Eigen::Infinity>()))) {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (VectorXs)x);
    } else {
      return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)x,
                            (VectorXs)u);
    }
  }

  void calcDiff(const std::shared_ptr<ActionData>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    if (std::isnan(
            scalar_cast<ScalarType>(u.template lpNorm<Eigen::Infinity>()))) {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (VectorXs)x);
    } else {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (VectorXs)x, (VectorXs)u);
    }
  }

  std::shared_ptr<ActionData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ActionData>>(createData.ptr());
    }
    return ActionModel::createData();
  }

  std::shared_ptr<ActionData> default_createData() {
    return this->ActionModel::createData();
  }

  void quasiStatic(const std::shared_ptr<ActionData>& data,
                   Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                   const std::size_t maxiter, const Scalar tol) override {
    if (boost::python::override quasiStatic =
            this->get_override("quasiStatic")) {
      u = bp::call<VectorXs>(quasiStatic.ptr(), data, (VectorXs)x, maxiter,
                             tol);
      if (static_cast<std::size_t>(u.size()) != nu_) {
        throw_pretty(
            "Invalid argument: " << "u has wrong dimension (it should be " +
                                        std::to_string(nu_) + ")");
      }
      return;
    }
    return ActionModel::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const std::shared_ptr<ActionData>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter, const Scalar tol) {
    return this->ActionModel::quasiStatic(data, u, x, maxiter, tol);
  }

  std::size_t get_ng() const override {
    if (boost::python::override get_ng = this->get_override("get_ng")) {
      return bp::call<std::size_t>(get_ng.ptr());
    }
    return this->ActionModel::get_ng();
  }

  std::size_t default_get_ng() const { return this->ActionModel::get_ng(); }

  std::size_t get_nh() const override {
    if (boost::python::override get_nh = this->get_override("get_nh")) {
      return bp::call<std::size_t>(get_nh.ptr());
    }
    return this->ActionModel::get_nh();
  }

  std::size_t default_get_nh() const { return this->ActionModel::get_nh(); }

  std::size_t get_ng_T() const override {
    if (boost::python::override get_ng_T = this->get_override("get_ng_T")) {
      return bp::call<std::size_t>(get_ng_T.ptr());
    }
    return this->ActionModel::get_ng_T();
  }

  std::size_t default_get_ng_T() const { return this->ActionModel::get_ng_T(); }

  std::size_t get_nh_T() const override {
    if (boost::python::override get_nh_T = this->get_override("get_nh_T")) {
      return bp::call<std::size_t>(get_nh_T.ptr());
    }
    return this->ActionModel::get_nh_T();
  }

  std::size_t default_get_nh_T() const { return this->ActionModel::get_nh_T(); }

  template <typename NewScalar>
  ActionModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ActionModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(state_->template cast<NewScalar>(), nu_, nr_, ng_, nh_,
                   ng_T_, nh_T_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
