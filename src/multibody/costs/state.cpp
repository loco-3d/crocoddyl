///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const Eigen::VectorXd& xref,
                               const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const Eigen::VectorXd& xref)
    : CostModelAbstract(state, activation), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& xref,
                               const std::size_t& nu)
    : CostModelAbstract(state, state->get_ndx(), nu), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& xref)
    : CostModelAbstract(state, state->get_ndx()), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state, const std::size_t& nu)
    : CostModelAbstract(state, state->get_ndx(), nu), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation)
    : CostModelAbstract(state, activation), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::CostModelState(boost::shared_ptr<StateMultibody> state)
    : CostModelAbstract(state, state->get_ndx()), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

CostModelState::~CostModelState() {}

void CostModelState::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                          const Eigen::Ref<const Eigen::VectorXd>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  state_->diff(xref_, x, data->r);
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelState::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

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

boost::shared_ptr<CostDataAbstract> CostModelState::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataState>(this, data);
}

const Eigen::VectorXd& CostModelState::get_xref() const { return xref_; }

void CostModelState::set_xref(const Eigen::VectorXd& xref_in) {
  if (static_cast<std::size_t>(xref_in.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  xref_ = xref_in;
}

}  // namespace crocoddyl
