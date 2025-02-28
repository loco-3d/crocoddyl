///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class DifferentialActionModelAbstractTpl_wrap
    : public DifferentialActionModelAbstractTpl<_Scalar>,
      public bp::wrapper<DifferentialActionModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DifferentialActionModelBase,
                         DifferentialActionModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef typename crocoddyl::DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModel;
  typedef typename crocoddyl::DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionData;
  typedef typename crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename DifferentialActionModel::VectorXs VectorXs;
  using DifferentialActionModel::ng_;
  using DifferentialActionModel::ng_T_;
  using DifferentialActionModel::nh_;
  using DifferentialActionModel::nh_T_;
  using DifferentialActionModel::nr_;
  using DifferentialActionModel::nu_;
  using DifferentialActionModel::state_;
  using DifferentialActionModel::unone_;

  DifferentialActionModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                          const std::size_t nu,
                                          const std::size_t nr = 1,
                                          const std::size_t ng = 0,
                                          const std::size_t nh = 0,
                                          const std::size_t ng_T = 0,
                                          const std::size_t nh_T = 0)
      : DifferentialActionModel(state, nu, nr, ng, nh, ng_T, nh_T),
        bp::wrapper<DifferentialActionModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  void calc(const std::shared_ptr<DifferentialActionData>& data,
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

  void calcDiff(const std::shared_ptr<DifferentialActionData>& data,
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

  std::shared_ptr<DifferentialActionData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<DifferentialActionData>>(
          createData.ptr());
    }
    return DifferentialActionModel::createData();
  }

  std::shared_ptr<DifferentialActionData> default_createData() {
    return this->DifferentialActionModel::createData();
  }

  void quasiStatic(const std::shared_ptr<DifferentialActionData>& data,
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
    return DifferentialActionModel::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const std::shared_ptr<DifferentialActionData>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter, const Scalar tol) {
    return this->DifferentialActionModel::quasiStatic(data, u, x, maxiter, tol);
  }

  template <typename NewScalar>
  DifferentialActionModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef DifferentialActionModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(state_->template cast<NewScalar>(), nr_, nu_, ng_, nh_,
                   ng_T_, nh_T_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
