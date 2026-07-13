///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DYNAMICS_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DYNAMICS_BASE_HPP_

#include "crocoddyl/core/dynamics-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class DynamicsModelAbstractTpl_wrap
    : public DynamicsModelAbstractTpl<_Scalar>,
      public bp::wrapper<DynamicsModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DynamicsModelBase, DynamicsModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef typename crocoddyl::DynamicsModelAbstractTpl<Scalar> DynamicsModel;
  typedef typename crocoddyl::DynamicsDataAbstractTpl<Scalar> DynamicsData;
  typedef typename crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename DynamicsModel::ParameterManager ParameterManager;
  typedef typename DynamicsModel::VectorXs VectorXs;
  using DynamicsModel::calc;
  using DynamicsModel::calcDiff;
  using DynamicsModel::calcDiff_xu;
  using DynamicsModel::createData;
  using DynamicsModel::ng_;
  using DynamicsModel::nh_;
  using DynamicsModel::np_;
  using DynamicsModel::nu_;
  using DynamicsModel::state_;
  using DynamicsModel::unone_;

  DynamicsModelAbstractTpl_wrap(
      std::shared_ptr<State> state,
      const DynamicsType dyn_type = DynamicsType::ContinuousControl,
      const std::size_t np = 0, const std::size_t nu = 0,
      const std::size_t ng = 0, const std::size_t nh = 0)
      : DynamicsModel(state, dyn_type, np, nu, ng, nh),
        bp::wrapper<DynamicsModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  void calc(const std::shared_ptr<DynamicsData>& data,
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

  void calcDiff_xu(const std::shared_ptr<DynamicsData>& data,
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
      return bp::call<void>(this->get_override("calcDiff_xu").ptr(), data,
                            (VectorXs)x);
    } else {
      return bp::call<void>(this->get_override("calcDiff_xu").ptr(), data,
                            (VectorXs)x, (VectorXs)u);
    }
  }

  void calcDiff_p(const std::shared_ptr<DynamicsData>& data,
                  const Eigen::Ref<const VectorXs>& x,
                  const Eigen::Ref<const VectorXs>& u) override {
    if (boost::python::override calcDiff_p = this->get_override("calcDiff_p")) {
      return bp::call<void>(calcDiff_p.ptr(), data, (VectorXs)x, (VectorXs)u);
    }
    return DynamicsModel::calcDiff_p(data, x, u);
  }

  void default_calcDiff_p(const std::shared_ptr<DynamicsData>& data,
                          const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u) {
    return this->DynamicsModel::calcDiff_p(data, x, u);
  }

  std::shared_ptr<DynamicsData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<DynamicsData>>(createData.ptr());
    }
    return DynamicsModel::createData();
  }

  std::shared_ptr<DynamicsData> default_createData() {
    return this->DynamicsModel::createData();
  }

  void set_params(const std::shared_ptr<DynamicsData>& data,
                  std::shared_ptr<ParameterManager> params) override {
    if (boost::python::override set_params = this->get_override("set_params")) {
      return bp::call<void>(set_params.ptr(), data, params);
    }
    return DynamicsModel::set_params(data, params);
  }

  void update_p(const std::shared_ptr<DynamicsData>& data,
                const Eigen::Ref<const VectorXs>& p) override {
    if (boost::python::override update_p = this->get_override("update_p")) {
      return bp::call<void>(update_p.ptr(), data, (VectorXs)p);
    }
    return DynamicsModel::update_p(data, p);
  }

  void quasiStatic(const std::shared_ptr<DynamicsData>& data,
                   Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                   const std::size_t maxiter, const Scalar tol) override {
    if (boost::python::override quasiStatic =
            this->get_override("quasiStatic")) {
      const VectorXs result = bp::call<VectorXs>(quasiStatic.ptr(), data,
                                                 (VectorXs)x, maxiter, tol);
      if (static_cast<std::size_t>(result.size()) != nu_) {
        throw_pretty(
            "Invalid argument: " << "u has wrong dimension (it should be " +
                                        std::to_string(nu_) + ")");
      }
      u = result;
      return;
    }
    return DynamicsModel::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const std::shared_ptr<DynamicsData>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter, const Scalar tol) {
    return this->DynamicsModel::quasiStatic(data, u, x, maxiter, tol);
  }

  template <typename NewScalar>
  DynamicsModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef DynamicsModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(state_->template cast<NewScalar>(), this->get_dyn_type(),
                   np_, nu_, ng_, nh_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DYNAMICS_BASE_HPP_
