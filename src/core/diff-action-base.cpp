///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

DifferentialActionModelAbstract::DifferentialActionModelAbstract(boost::shared_ptr<StateAbstract> state,
                                                                 const std::size_t& nu, const std::size_t& nr)
    : nu_(nu),
      nr_(nr),
      state_(state),
      unone_(Eigen::VectorXd::Zero(nu)),
      u_lb_(Eigen::VectorXd::Constant(nu, -std::numeric_limits<double>::infinity())),
      u_ub_(Eigen::VectorXd::Constant(nu, std::numeric_limits<double>::infinity())),
      has_control_limits_(false) {}

DifferentialActionModelAbstract::~DifferentialActionModelAbstract() {}

void DifferentialActionModelAbstract::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void DifferentialActionModelAbstract::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                               const Eigen::Ref<const Eigen::VectorXd>& x, const bool&) {
  calcDiff(data, x, unone_);
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelAbstract::createData() {
  return boost::make_shared<DifferentialActionDataAbstract>(this);
}

const std::size_t& DifferentialActionModelAbstract::get_nu() const { return nu_; }

const std::size_t& DifferentialActionModelAbstract::get_nr() const { return nr_; }

const boost::shared_ptr<StateAbstract>& DifferentialActionModelAbstract::get_state() const { return state_; }

const Eigen::VectorXd& DifferentialActionModelAbstract::get_u_lb() const { return u_lb_; }

const Eigen::VectorXd& DifferentialActionModelAbstract::get_u_ub() const { return u_ub_; }

bool const& DifferentialActionModelAbstract::get_has_control_limits() const { return has_control_limits_; }

void DifferentialActionModelAbstract::set_u_lb(const Eigen::VectorXd& u_lb) {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_lb_ = u_lb;
  update_has_control_limits();
}

void DifferentialActionModelAbstract::set_u_ub(const Eigen::VectorXd& u_ub) {
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_ub_ = u_ub;
  update_has_control_limits();
}

void DifferentialActionModelAbstract::update_has_control_limits() {
  has_control_limits_ = isfinite(u_lb_.array()).any() && isfinite(u_ub_.array()).any();
}

}  // namespace crocoddyl
