///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {

CostModelState::CostModelState(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::VectorXd& xref,
                               unsigned int const& nu)
    : CostModelAbstract(state, activation, nu), xref_(xref) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::VectorXd& xref)
    : CostModelAbstract(state, activation), xref_(xref) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state, const Eigen::VectorXd& xref, unsigned int const& nu)
    : CostModelAbstract(state, state.get_ndx(), nu), xref_(xref) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state, const Eigen::VectorXd& xref)
    : CostModelAbstract(state, state.get_ndx()), xref_(xref) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state, ActivationModelAbstract& activation, unsigned int const& nu)
    : CostModelAbstract(state, activation, nu), xref_(state.zero()) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state, unsigned int const& nu)
    : CostModelAbstract(state, state.get_ndx(), nu), xref_(state.zero()) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state, ActivationModelAbstract& activation)
    : CostModelAbstract(state, activation), xref_(state.zero()) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(StateMultibody& state)
    : CostModelAbstract(state, state.get_ndx()), xref_(state.zero()) {
  assert(xref_.size() == state_.get_nx() && "CostModelState: reference is not dimension nx");
  assert(activation_.get_nr() == state_.get_ndx() && "CostModelState: nr is not equals to ndx");
}

CostModelState::~CostModelState() {}

void CostModelState::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  assert(x.size() == state_.get_nx() && "CostModelState::calc: x has wrong dimension");

  state_.diff(xref_, x, data->r);
  activation_.calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelState::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  assert(x.size() == state_.get_nx() && "CostModelState::calcDiff: x has wrong dimension");

  if (recalc) {
    calc(data, x, u);
  }
  state_.Jdiff(xref_, x, data->Rx, data->Rx, second);
  activation_.calcDiff(data->activation, data->r, recalc);
  data->Lx = data->Rx.transpose() * data->activation->Ar;
  data->Lxx = data->Rx.transpose() * data->activation->Arr * data->Rx;
}

const Eigen::VectorXd& CostModelState::get_xref() const { return xref_; }

}  // namespace crocoddyl
