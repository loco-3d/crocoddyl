///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CONSTRAINT_BASE_HPP_

#include "crocoddyl/core/constraint-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class ConstraintModelAbstractTpl_wrap
    : public ConstraintModelAbstractTpl<_Scalar>,
      public bp::wrapper<ConstraintModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ConstraintModelBase, ConstraintModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef typename crocoddyl::ConstraintModelAbstractTpl<Scalar>
      ConstraintModel;
  typedef typename crocoddyl::ConstraintDataAbstractTpl<Scalar> ConstraintData;
  typedef typename ConstraintModel::StateAbstract State;
  typedef typename ConstraintModel::ResidualModelAbstract ResidualModel;
  typedef typename ConstraintModel::DataCollectorAbstract DataCollector;
  typedef typename ConstraintModel::VectorXs VectorXs;
  using ConstraintModel::ng_;
  using ConstraintModel::nh_;
  using ConstraintModel::nu_;
  using ConstraintModel::residual_;
  using ConstraintModel::state_;
  using ConstraintModel::T_constraint_;
  using ConstraintModel::unone_;

  ConstraintModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                  std::shared_ptr<ResidualModel> residual,
                                  const std::size_t ng, const std::size_t nh)
      : ConstraintModel(state, residual, ng, nh),
        bp::wrapper<ConstraintModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  ConstraintModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                  const std::size_t nu, const std::size_t ng,
                                  const std::size_t nh,
                                  const bool T_const = true)
      : ConstraintModel(state, nu, ng, nh, T_const),
        bp::wrapper<ConstraintModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  ConstraintModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                  const std::size_t ng, const std::size_t nh,
                                  const bool T_const = true)
      : ConstraintModel(state, ng, nh, T_const) {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  void calc(const std::shared_ptr<ConstraintData>& data,
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

  void calcDiff(const std::shared_ptr<ConstraintData>& data,
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

  std::shared_ptr<ConstraintData> createData(
      DataCollector* const data) override {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ConstraintData>>(createData.ptr(),
                                                       boost::ref(data));
    }
    return ConstraintModel::createData(data);
  }

  std::shared_ptr<ConstraintData> default_createData(
      DataCollector* const data) {
    return this->ConstraintModel::createData(data);
  }

  template <typename NewScalar>
  ConstraintModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ConstraintModelAbstractTpl_wrap<NewScalar> ReturnType;
    if (residual_) {
      ReturnType ret(state_->template cast<NewScalar>(),
                     residual_->template cast<NewScalar>(), ng_, nh_);
      return ret;
    } else {
      ReturnType ret(state_->template cast<NewScalar>(), nu_, ng_, nh_,
                     T_constraint_);
      return ret;
    }
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
