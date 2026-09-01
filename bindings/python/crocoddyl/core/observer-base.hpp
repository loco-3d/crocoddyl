///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_OBSERVER_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_OBSERVER_BASE_HPP_

#include "crocoddyl/core/observer-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

/**
 * @brief Python wrapper for abstract observer models
 *
 * This wrapper follows the same conventions as ActionModelAbstractTpl_wrap and
 * adds observer-specific callbacks (e.g., parameter and measured-torque
 * updates).
 */
template <typename _Scalar>
class ObserverModelAbstractTpl_wrap
    : public ObserverModelAbstractTpl<_Scalar>,
      public bp::wrapper<ObserverModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ObserverModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef crocoddyl::ObserverModelAbstractTpl<Scalar> ObserverModel;
  typedef crocoddyl::ActionDataAbstractTpl<Scalar> ActionData;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename ObserverModel::VectorXs VectorXs;
  using ObserverModel::calc;
  using ObserverModel::calcDiff;
  using ObserverModel::createData;
  using ObserverModel::ng_;
  using ObserverModel::ng_T_;
  using ObserverModel::nh_;
  using ObserverModel::nh_T_;
  using ObserverModel::np_;
  using ObserverModel::nr_;
  using ObserverModel::ntau_;
  using ObserverModel::nu_;
  using ObserverModel::state_;
  using ObserverModel::tau_meas_;
  using ObserverModel::unone_;

  ObserverModelAbstractTpl_wrap(
      std::shared_ptr<State> state, const std::size_t ntau,
      const std::size_t nu, const std::size_t nr = 0, const std::size_t ng = 0,
      const std::size_t nh = 0, const std::size_t ng_T = 0,
      const std::size_t nh_T = 0, const std::size_t np = 0)
      : ObserverModel(state, ntau, nu, nr, ng, nh, ng_T, nh_T, np),
        bp::wrapper<ObserverModel>() {
    unone_ = VectorXs::Constant(nu, Scalar(NAN));
  }

  /**
   * @brief Call Python implementation of calc()
   */
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

  /**
   * @brief Call Python implementation of calcDiff()
   */
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

  void set_params(const std::shared_ptr<ActionData>& data,
                  std::shared_ptr<ParameterManager> params) override {
    if (boost::python::override set_params = this->get_override("set_params")) {
      return bp::call<void>(set_params.ptr(), data, params);
    }
    return ObserverModel::set_params(data, params);
  }

  void default_set_params(const std::shared_ptr<ActionData>& data,
                          std::shared_ptr<ParameterManager> params) {
    return this->ObserverModel::set_params(data, params);
  }

  void update_p(const std::shared_ptr<ActionData>& data,
                const Eigen::Ref<const VectorXs>& p) override {
    return bp::call<void>(this->get_override("update_p").ptr(), data,
                          (VectorXs)p);
  }

  void update_tau(const Eigen::Ref<const VectorXs>& tau_meas) override {
    if (boost::python::override update_tau = this->get_override("update_tau")) {
      return bp::call<void>(update_tau.ptr(), (VectorXs)tau_meas);
    }
    return ObserverModel::update_tau(tau_meas);
  }

  void default_update_tau(const Eigen::Ref<const VectorXs>& tau_meas) {
    return this->ObserverModel::update_tau(tau_meas);
  }

  /**
   * @brief Create observer data
   */
  std::shared_ptr<ActionData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ActionData>>(createData.ptr());
    }
    return ObserverModel::createData();
  }

  std::shared_ptr<ActionData> default_createData() {
    return this->ObserverModel::createData();
  }

  /**
   * @brief Compute quasic-static commands
   */
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
    return ObserverModel::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const std::shared_ptr<ActionData>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter, const Scalar tol) {
    return this->ObserverModel::quasiStatic(data, u, x, maxiter, tol);
  }

  /**
   * @brief Return the number of inequality constraints
   */
  std::size_t get_ng() const override {
    if (boost::python::override get_ng = this->get_override("get_ng")) {
      return bp::call<std::size_t>(get_ng.ptr());
    }
    return this->ObserverModel::get_ng();
  }

  std::size_t default_get_ng() const { return this->ObserverModel::get_ng(); }

  /**
   * @brief Return the number of equality constraints
   */
  std::size_t get_nh() const override {
    if (boost::python::override get_nh = this->get_override("get_nh")) {
      return bp::call<std::size_t>(get_nh.ptr());
    }
    return this->ObserverModel::get_nh();
  }

  std::size_t default_get_nh() const { return this->ObserverModel::get_nh(); }

  /**
   * @brief Return the number of terminal inequality constraints
   */
  std::size_t get_ng_T() const override {
    if (boost::python::override get_ng_T = this->get_override("get_ng_T")) {
      return bp::call<std::size_t>(get_ng_T.ptr());
    }
    return this->ObserverModel::get_ng_T();
  }

  std::size_t default_get_ng_T() const {
    return this->ObserverModel::get_ng_T();
  }

  /**
   * @brief Return the number of terminal equality constraints
   */
  std::size_t get_nh_T() const override {
    if (boost::python::override get_nh_T = this->get_override("get_nh_T")) {
      return bp::call<std::size_t>(get_nh_T.ptr());
    }
    return this->ObserverModel::get_nh_T();
  }

  std::size_t default_get_nh_T() const {
    return this->ObserverModel::get_nh_T();
  }

  /**
   * @brief Cast the wrapper to a different scalar type
   */
  template <typename NewScalar>
  ObserverModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ObserverModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(state_->template cast<NewScalar>(), ntau_, nu_, nr_, ng_,
                   nh_, ng_T_, nh_T_, np_);
    ret.update_tau(tau_meas_.template cast<NewScalar>());
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_OBSERVER_BASE_HPP_
