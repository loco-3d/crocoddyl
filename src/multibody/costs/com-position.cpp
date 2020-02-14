///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"

namespace crocoddyl {

CostModelCoMPosition::CostModelCoMPosition(boost::shared_ptr<StateMultibody> state,
                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                           const Eigen::Vector3d& cref, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), cref_(cref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelCoMPosition::CostModelCoMPosition(boost::shared_ptr<StateMultibody> state,
                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                           const Eigen::Vector3d& cref)
    : CostModelAbstract(state, activation), cref_(cref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

CostModelCoMPosition::CostModelCoMPosition(boost::shared_ptr<StateMultibody> state, const Eigen::Vector3d& cref,
                                           const std::size_t& nu)
    : CostModelAbstract(state, 3, nu), cref_(cref) {}

CostModelCoMPosition::CostModelCoMPosition(boost::shared_ptr<StateMultibody> state, const Eigen::Vector3d& cref)
    : CostModelAbstract(state, 3), cref_(cref) {}

CostModelCoMPosition::~CostModelCoMPosition() {}

void CostModelCoMPosition::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&) {
  // Compute the cost residual give the reference CoMPosition position
  CostDataCoMPosition* d = static_cast<CostDataCoMPosition*>(data.get());
  data->r = d->pinocchio->com[0] - cref_;

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelCoMPosition::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& x,
                                    const Eigen::Ref<const Eigen::VectorXd>& u) {
  CostDataCoMPosition* d = static_cast<CostDataCoMPosition*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r);
  data->Rx.leftCols(nv) = d->pinocchio->Jcom;
  data->Lx.head(nv).noalias() = d->pinocchio->Jcom.transpose() * data->activation->Ar;
  d->Arr_Jcom.noalias() = data->activation->Arr * d->pinocchio->Jcom;
  data->Lxx.topLeftCorner(nv, nv).noalias() = d->pinocchio->Jcom.transpose() * d->Arr_Jcom;
}

boost::shared_ptr<CostDataAbstract> CostModelCoMPosition::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataCoMPosition>(this, data);
}

const Eigen::Vector3d& CostModelCoMPosition::get_cref() const { return cref_; }

void CostModelCoMPosition::set_cref(const Eigen::Vector3d& cref_in) { cref_ = cref_in; }

}  // namespace crocoddyl
