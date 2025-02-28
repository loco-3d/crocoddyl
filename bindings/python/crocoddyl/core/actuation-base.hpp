
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class ActuationModelAbstractTpl_wrap
    : public ActuationModelAbstractTpl<Scalar>,
      public bp::wrapper<ActuationModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase, ActuationModelAbstractTpl_wrap)

  typedef typename crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModel;
  typedef typename crocoddyl::ActuationDataAbstractTpl<Scalar> ActuationData;
  typedef typename ActuationModel::StateAbstract State;
  typedef typename ActuationModel::VectorXs VectorXs;
  using ActuationModel::nu_;
  using ActuationModel::state_;

  ActuationModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                 const std::size_t nu)
      : ActuationModel(state, nu), bp::wrapper<ActuationModel>() {}

  void calc(const std::shared_ptr<ActuationData>& data,
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
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)x,
                          (VectorXs)u);
  }

  void calcDiff(const std::shared_ptr<ActuationData>& data,
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
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)x, (VectorXs)u);
  }

  void commands(const std::shared_ptr<ActuationData>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& tau) override {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(tau.size()) != state_->get_nv()) {
      throw_pretty(
          "Invalid argument: " << "tau has wrong dimension (it should be " +
                                      std::to_string(state_->get_nv()) + ")");
    }
    return bp::call<void>(this->get_override("commands").ptr(), data,
                          (VectorXs)x, (VectorXs)tau);
  }

  void torqueTransform(const std::shared_ptr<ActuationData>& data,
                       const Eigen::Ref<const VectorXs>& x,
                       const Eigen::Ref<const VectorXs>& u) override {
    if (boost::python::override torqueTransform =
            this->get_override("torqueTransform")) {
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
      return bp::call<void>(torqueTransform.ptr(), data, (VectorXs)x,
                            (VectorXs)u);
    }
    return ActuationModel::torqueTransform(data, x, u);
  }

  void default_torqueTransform(const std::shared_ptr<ActuationData>& data,
                               const Eigen::Ref<const VectorXs>& x,
                               const Eigen::Ref<const VectorXs>& u) {
    return this->ActuationModel::torqueTransform(data, x, u);
  }

  std::shared_ptr<ActuationData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ActuationData>>(createData.ptr());
    }
    return ActuationModel::createData();
  }

  std::shared_ptr<ActuationData> default_createData() {
    return this->ActuationModel::createData();
  }

  template <typename NewScalar>
  ActuationModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ActuationModelAbstractTpl_wrap<NewScalar> ReturnType;
    typedef StateAbstractTpl<NewScalar> StateType;
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        nu_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_
