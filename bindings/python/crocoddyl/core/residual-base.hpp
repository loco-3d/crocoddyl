///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class ResidualModelAbstractTpl_wrap
    : public ResidualModelAbstractTpl<_Scalar>,
      public bp::wrapper<ResidualModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef typename ScalarSelector<Scalar>::type ScalarType;
  typedef typename crocoddyl::ResidualModelAbstractTpl<Scalar> ResidualModel;
  typedef typename crocoddyl::ResidualDataAbstractTpl<Scalar> ResidualData;
  typedef typename crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename ResidualModel::CostDataAbstract CostData;
  typedef typename ResidualModel::ActivationDataAbstract ActivationData;
  typedef typename ResidualModel::DataCollectorAbstract DataCollectorAbstract;
  typedef typename ResidualModel::VectorXs VectorXs;
  using ResidualModel::nr_;
  using ResidualModel::nu_;
  using ResidualModel::q_dependent_;
  using ResidualModel::state_;
  using ResidualModel::u_dependent_;
  using ResidualModel::unone_;
  using ResidualModel::v_dependent_;

  ResidualModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                const std::size_t nr, const std::size_t nu,
                                const bool q_dependent = true,
                                const bool v_dependent = true,
                                const bool u_dependent = true)
      : ResidualModel(state, nr, nu, q_dependent, v_dependent, u_dependent),
        bp::wrapper<ResidualModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  ResidualModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                const std::size_t nr,
                                const bool q_dependent = true,
                                const bool v_dependent = true,
                                const bool u_dependent = true)
      : ResidualModel(state, nr, q_dependent, v_dependent, u_dependent),
        bp::wrapper<ResidualModel>() {
    unone_ = VectorXs::Constant(nu_, Scalar(NAN));
  }

  void calc(const std::shared_ptr<ResidualData>& data,
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

  void calcDiff(const std::shared_ptr<ResidualData>& data,
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

  std::shared_ptr<ResidualData> createData(
      DataCollectorAbstract* const data) override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ResidualData>>(createData.ptr(),
                                                     boost::ref(data));
    }
    return ResidualModel::createData(data);
  }

  std::shared_ptr<ResidualData> default_createData(
      DataCollectorAbstract* const data) {
    return this->ResidualModel::createData(data);
  }

  void calcCostDiff(const std::shared_ptr<CostData>& cdata,
                    const std::shared_ptr<ResidualData>& rdata,
                    const std::shared_ptr<ActivationData>& adata,
                    const bool update_u = true) override {
    if (boost::python::override calcCostDiff =
            this->get_override("calcCostDiff")) {
      return bp::call<void>(calcCostDiff.ptr(), boost::ref(cdata),
                            boost::ref(rdata), boost::ref(adata), update_u);
    }
    return ResidualModel::calcCostDiff(cdata, rdata, adata, update_u);
  }

  void default_calcCostDiff(const std::shared_ptr<CostData>& cdata,
                            const std::shared_ptr<ResidualData>& rdata,
                            const std::shared_ptr<ActivationData>& adata,
                            const bool update_u) {
    return this->ResidualModel::calcCostDiff(cdata, rdata, adata, update_u);
  }

  void default_calcCostDiff_noupdate_u(
      const std::shared_ptr<CostData>& cdata,
      const std::shared_ptr<ResidualData>& rdata,
      const std::shared_ptr<ActivationData>& adata) {
    return this->ResidualModel::calcCostDiff(cdata, rdata, adata);
  }

  template <typename NewScalar>
  ResidualModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ResidualModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(state_->template cast<NewScalar>(), nr_, nu_, q_dependent_,
                   v_dependent_, u_dependent_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_
