#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state,
                               ActivationModelAbstract* const activation, const Eigen::VectorXd& xref,
                               const unsigned int& nu)
    : CostModelAbstract(model, activation, nu), state_(state), xref_(xref) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state,
                               ActivationModelAbstract* const activation, const Eigen::VectorXd& xref)
    : CostModelAbstract(model, activation), state_(state), xref_(xref) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state, const Eigen::VectorXd& xref,
                               const unsigned int& nu)
    : CostModelAbstract(model, state->get_ndx(), nu), state_(state), xref_(xref) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state, const Eigen::VectorXd& xref)
    : CostModelAbstract(model, state->get_ndx()), state_(state), xref_(xref) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state,
                               ActivationModelAbstract* const activation, const unsigned int& nu)
    : CostModelAbstract(model, activation, nu), state_(state), xref_(state->zero()) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state, const unsigned int& nu)
    : CostModelAbstract(model, state->get_ndx(), nu), state_(state), xref_(state->zero()) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state,
                               ActivationModelAbstract* const activation)
    : CostModelAbstract(model, activation), state_(state), xref_(state->zero()) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state)
    : CostModelAbstract(model, state->get_ndx()), state_(state), xref_(state->zero()) {
  assert(xref_.size() == nx_ && "CostModelState: reference is not dimension nx");
  assert(nr_ == ndx_ && "CostModelState: nr is not equals to ndx");
}

CostModelState::~CostModelState() {}

void CostModelState::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  assert(x.size() == nx_ && "CostModelState::calc: x has wrong dimension");

  state_->diff(xref_, x, data->r);
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelState::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  assert(x.size() == nx_ && "CostModelState::calcDiff: x has wrong dimension");

  if (recalc) {
    calc(data, x, u);
  }
  state_->Jdiff(xref_, x, data->Rx, data->Rx, second);
  activation_->calcDiff(data->activation, data->r, recalc);
  data->Lx = data->Rx.transpose() * data->activation->Ar;
  data->Lxx = data->Rx.transpose() * data->activation->Arr * data->Rx;
}

StateAbstract* CostModelState::get_state() const { return state_; }

const Eigen::VectorXd& CostModelState::get_xref() const { return xref_; }

}  // namespace crocoddyl