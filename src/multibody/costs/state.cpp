///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const Eigen::VectorXd& xref,
                               const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), xref_(xref) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const Eigen::VectorXd& xref)
    : CostModelAbstract(state, activation), xref_(xref) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& xref,
                               const std::size_t& nu)
    : CostModelAbstract(state, state->get_ndx(), nu), xref_(xref) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& xref)
    : CostModelAbstract(state, state->get_ndx()), xref_(xref) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), xref_(state->zero()) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state, const std::size_t& nu)
    : CostModelAbstract(state, state->get_ndx(), nu), xref_(state->zero()) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation)
    : CostModelAbstract(state, activation), xref_(state->zero()) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state)
    : CostModelAbstract(state, state->get_ndx()), xref_(state->zero()) {
  assert(static_cast<std::size_t>(xref_.size()) == state_->get_nx() && "reference is not dimension nx");
  assert(activation_->get_nr() == state_->get_ndx() && "nr is not equals to ndx");
}

CostModelState::~CostModelState() {}

void CostModelState::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  assert(static_cast<std::size_t>(x.size()) == state_->get_nx() && "x has wrong dimension");

  state_->diff(xref_, x, data->r);
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelState::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  assert(static_cast<std::size_t>(x.size()) == state_->get_nx() && "x has wrong dimension");

  CostDataState* d = static_cast<CostDataState*>(data.get());
  if (recalc) {
    calc(data, x, u);
  }
  state_->Jdiff(xref_, x, data->Rx, data->Rx, second);
  activation_->calcDiff(data->activation, data->r, recalc);
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

boost::shared_ptr<CostDataAbstract> CostModelState::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataState>(this, data);
}

const Eigen::VectorXd& CostModelState::get_xref() const { return xref_; }

}  // namespace crocoddyl
