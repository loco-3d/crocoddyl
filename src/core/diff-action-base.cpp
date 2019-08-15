///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

DifferentialActionModelAbstract::DifferentialActionModelAbstract(StateAbstract& state, unsigned int const& nu,
                                                                 unsigned int const& nr)
    : nu_(nu), nr_(nr), state_(state), unone_(Eigen::VectorXd::Zero(nu)) {
  assert(nu_ != 0 && "DifferentialActionModelAbstract: nu cannot be zero");
  assert(nr_ != 0 && "DifferentialActionModelAbstract: nr cannot be zero");
}

DifferentialActionModelAbstract::~DifferentialActionModelAbstract() {}

void DifferentialActionModelAbstract::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void DifferentialActionModelAbstract::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelAbstract::createData() {
  return boost::make_shared<DifferentialActionDataAbstract>(this);
}

const unsigned int& DifferentialActionModelAbstract::get_nu() const { return nu_; }

const unsigned int& DifferentialActionModelAbstract::get_nr() const { return nr_; }

StateAbstract& DifferentialActionModelAbstract::get_state() const { return state_; }

}  // namespace crocoddyl
