///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/control.hpp"

namespace crocoddyl {

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ActivationModelAbstract> activation, const Eigen::VectorXd& uref)
    : CostModelAbstract(state, activation, static_cast<std::size_t>(uref.size())), uref_(uref) {
  assert(activation.get_nr() == static_cast<std::size_t>(uref.size()) && "nr is not equals to nu");
}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ActivationModelAbstract> activation)
    : CostModelAbstract(state, activation), uref_(Eigen::VectorXd::Zero(activation->get_nr())) {}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), uref_(Eigen::VectorXd::Zero(nu)) {
  assert(activation.get_nr() == nu_ && "nr is not equals to nu");
}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& uref)
    : CostModelAbstract(state, static_cast<std::size_t>(uref.size()), static_cast<std::size_t>(uref.size())),
      uref_(uref) {}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state)
    : CostModelAbstract(state, state->get_nv()), uref_(Eigen::VectorXd::Zero(state->get_nv())) {}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu)
    : CostModelAbstract(state, nu, nu), uref_(Eigen::VectorXd::Zero(nu)) {}

CostModelControl::~CostModelControl() {}

void CostModelControl::calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>&,
                            const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(nu_ != 0 && "it seems to be an autonomous system, if so, don't add this cost function");
  assert(static_cast<std::size_t>(u.size()) == nu_ && "u has wrong dimension");

  data->r = u - uref_;
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelControl::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                                const bool& recalc) {
  assert(nu_ != 0 && "it seems to be an autonomous system, if so, don't add this cost function");
  assert(static_cast<std::size_t>(u.size()) == nu_ && "u has wrong dimension");

  if (recalc) {
    calc(data, x, u);
  }
  activation_->calcDiff(data->activation, data->r, recalc);
  data->Lu = data->activation->Ar;
  data->Luu.diagonal() = data->activation->Arr.diagonal();
}

const Eigen::VectorXd& CostModelControl::get_uref() const { return uref_; }

}  // namespace crocoddyl
