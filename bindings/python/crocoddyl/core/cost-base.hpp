///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_COST_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_COST_BASE_HPP_

#include "crocoddyl/core/cost-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class CostModelAbstractTpl_wrap
    : public CostModelAbstractTpl<_Scalar>,
      public bp::wrapper<CostModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(CostModelBase, CostModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModel;
  typedef typename crocoddyl::CostDataAbstractTpl<Scalar> CostData;
  typedef typename CostModel::StateAbstract State;
  typedef typename CostModel::ActivationModelAbstract ActivationModel;
  typedef typename CostModel::ResidualModelAbstract ResidualModel;
  typedef typename CostModel::DataCollectorAbstract DataCollector;
  typedef typename CostModel::VectorXs VectorXs;
  using CostModel::activation_;
  using CostModel::nu_;
  using CostModel::residual_;
  using CostModel::state_;
  using CostModel::unone_;

  CostModelAbstractTpl_wrap(std::shared_ptr<State> state,
                            std::shared_ptr<ActivationModel> activation,
                            std::shared_ptr<ResidualModel> residual)
      : CostModel(state, activation, residual), bp::wrapper<CostModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  CostModelAbstractTpl_wrap(std::shared_ptr<State> state,
                            std::shared_ptr<ActivationModel> activation,
                            const std::size_t nu)
      : CostModel(state, activation, nu), bp::wrapper<CostModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  CostModelAbstractTpl_wrap(std::shared_ptr<State> state,
                            std::shared_ptr<ActivationModel> activation)
      : CostModel(state, activation), bp::wrapper<CostModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  CostModelAbstractTpl_wrap(std::shared_ptr<State> state,
                            std::shared_ptr<ResidualModel> residual)
      : CostModel(state, residual), bp::wrapper<CostModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  CostModelAbstractTpl_wrap(std::shared_ptr<State> state, const std::size_t nr,
                            const std::size_t nu)
      : CostModel(state, nr, nu), bp::wrapper<CostModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  CostModelAbstractTpl_wrap(std::shared_ptr<State> state, const std::size_t nr)
      : CostModel(state, nr), bp::wrapper<CostModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  void calc(const std::shared_ptr<CostData>& data,
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

  void calcDiff(const std::shared_ptr<CostData>& data,
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

  std::shared_ptr<CostData> createData(DataCollector* const data) override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<CostData>>(createData.ptr(),
                                                 boost::ref(data));
    }
    return CostModel::createData(data);
  }

  std::shared_ptr<CostData> default_createData(DataCollector* const data) {
    return this->CostModel::createData(data);
  }

  template <typename NewScalar>
  CostModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef CostModelAbstractTpl_wrap<NewScalar> ReturnType;
    if (residual_) {
      ReturnType ret(state_->template cast<NewScalar>(),
                     activation_->template cast<NewScalar>(),
                     residual_->template cast<NewScalar>());
      return ret;
    } else {
      ReturnType ret(state_->template cast<NewScalar>(),
                     activation_->template cast<NewScalar>(), nu_);
      return ret;
    }
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
