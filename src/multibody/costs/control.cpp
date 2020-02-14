///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/control.hpp"

namespace crocoddyl {

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ActivationModelAbstract> activation, const Eigen::VectorXd& uref)
    : CostModelAbstract(state, activation, static_cast<std::size_t>(uref.size())), uref_(uref) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ActivationModelAbstract> activation)
    : CostModelAbstract(state, activation), uref_(Eigen::VectorXd::Zero(activation->get_nr())) {}

CostModelControl::CostModelControl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), uref_(Eigen::VectorXd::Zero(nu)) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
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
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  data->r = u - uref_;
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelControl::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  activation_->calcDiff(data->activation, data->r);
  data->Lu = data->activation->Ar;
  data->Luu.diagonal() = data->activation->Arr.diagonal();
}

const Eigen::VectorXd& CostModelControl::get_uref() const { return uref_; }

void CostModelControl::set_uref(const Eigen::VectorXd& uref_in) {
  if (static_cast<std::size_t>(uref_in.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "uref has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  uref_ = uref_in;
}

}  // namespace crocoddyl
