///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2024, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class ResidualModelAbstract_wrap : public ResidualModelAbstract,
                                   public bp::wrapper<ResidualModelAbstract> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using ResidualModelAbstract::nu_;
  using ResidualModelAbstract::unone_;

  ResidualModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                             const std::size_t nr, const std::size_t nu,
                             const bool q_dependent = true,
                             const bool v_dependent = true,
                             const bool u_dependent = true)
      : ResidualModelAbstract(state, nr, nu, q_dependent, v_dependent,
                              u_dependent),
        bp::wrapper<ResidualModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu);
  }

  ResidualModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                             const std::size_t nr,
                             const bool q_dependent = true,
                             const bool v_dependent = true,
                             const bool u_dependent = true)
      : ResidualModelAbstract(state, nr, q_dependent, v_dependent, u_dependent),
        bp::wrapper<ResidualModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
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
    if (std::isnan(u.lpNorm<Eigen::Infinity>())) {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (Eigen::VectorXd)x);
    } else {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)u);
    }
  }

  void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
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
    if (std::isnan(u.lpNorm<Eigen::Infinity>())) {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (Eigen::VectorXd)x);
    } else {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)u);
    }
  }

  std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ResidualDataAbstract> >(createData.ptr(),
                                                              boost::ref(data));
    }
    return ResidualModelAbstract::createData(data);
  }

  std::shared_ptr<ResidualDataAbstract> default_createData(
      DataCollectorAbstract* const data) {
    return this->ResidualModelAbstract::createData(data);
  }

  void calcCostDiff(const std::shared_ptr<CostDataAbstract>& cdata,
                    const std::shared_ptr<ResidualDataAbstract>& rdata,
                    const std::shared_ptr<ActivationDataAbstract>& adata,
                    const bool update_u = true) {
    if (boost::python::override calcCostDiff =
            this->get_override("calcCostDiff")) {
      return bp::call<void>(calcCostDiff.ptr(), boost::ref(cdata),
                            boost::ref(rdata), boost::ref(adata), update_u);
    }
    return ResidualModelAbstract::calcCostDiff(cdata, rdata, adata, update_u);
  }

  void default_calcCostDiff(
      const std::shared_ptr<CostDataAbstract>& cdata,
      const std::shared_ptr<ResidualDataAbstract>& rdata,
      const std::shared_ptr<ActivationDataAbstract>& adata,
      const bool update_u) {
    return this->ResidualModelAbstract::calcCostDiff(cdata, rdata, adata,
                                                     update_u);
  }

  void default_calcCostDiff_noupdate_u(
      const std::shared_ptr<CostDataAbstract>& cdata,
      const std::shared_ptr<ResidualDataAbstract>& rdata,
      const std::shared_ptr<ActivationDataAbstract>& adata) {
    return this->ResidualModelAbstract::calcCostDiff(cdata, rdata, adata);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_
