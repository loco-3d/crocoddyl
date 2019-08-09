///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

ActionModelAbstract::ActionModelAbstract(StateAbstract& state, const unsigned int& nu, const unsigned int& nr)
    : nu_(nu), nr_(nr), state_(state), unone_(Eigen::VectorXd::Zero(nu)) {}

ActionModelAbstract::~ActionModelAbstract() {}

void ActionModelAbstract::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void ActionModelAbstract::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

const unsigned int& ActionModelAbstract::get_nu() const { return nu_; }

const unsigned int& ActionModelAbstract::get_nr() const { return nr_; }

StateAbstract& ActionModelAbstract::get_state() const { return state_; }

}  // namespace crocoddyl
