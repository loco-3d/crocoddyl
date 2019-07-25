#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state,
                               ActivationModelAbstract* const activation, const Eigen::VectorXd& xref,
                               const unsigned int& nu)
    : CostModelAbstract(model, activation, state->get_ndx(), nu), state_(state), xref_(xref) {}

CostModelState::CostModelState(pinocchio::Model* const model, StateAbstract* state,
                               const Eigen::VectorXd& xref, const unsigned int& nu)
    : CostModelAbstract(model, state->get_ndx(), nu), state_(state), xref_(xref) {}

CostModelState::~CostModelState() {}

void CostModelState::calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  state_->diff(xref_, x, data->r);
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelState::calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  state_->Jdiff(xref_, x, data->Rx, data->Rx, second);
  activation_->calcDiff(data->activation, data->r, recalc);
  data->Lx = data->Rx.transpose() * data->activation->Ar;
  data->Lxx = data->Rx.transpose() * data->activation->Arr * data->Rx;
}

StateAbstract* CostModelState::get_state() const { return state_; }

}  // namespace crocoddyl