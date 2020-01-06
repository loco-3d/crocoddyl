///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

StateAbstract::StateAbstract(const std::size_t& nx, const std::size_t& ndx)
    : nx_(nx),
      ndx_(ndx),
      lb_(Eigen::VectorXd::Constant(nx_, -std::numeric_limits<double>::infinity())),
      ub_(Eigen::VectorXd::Constant(nx_, std::numeric_limits<double>::infinity())),
      has_limits_(false) {
  nv_ = ndx / 2;
  nq_ = nx_ - nv_;
}

StateAbstract::~StateAbstract() {}

const std::size_t& StateAbstract::get_nx() const { return nx_; }

const std::size_t& StateAbstract::get_ndx() const { return ndx_; }

const std::size_t& StateAbstract::get_nq() const { return nq_; }

const std::size_t& StateAbstract::get_nv() const { return nv_; }

const Eigen::VectorXd& StateAbstract::get_lb() const { return lb_; }

const Eigen::VectorXd& StateAbstract::get_ub() const { return ub_; }

bool const& StateAbstract::get_has_limits() const { return has_limits_; }

void StateAbstract::set_lb(const Eigen::VectorXd& lb) {
  if (static_cast<std::size_t>(lb.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  lb_ = lb;
  update_has_limits();
}

void StateAbstract::set_ub(const Eigen::VectorXd& ub) {
  if (static_cast<std::size_t>(ub.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  ub_ = ub;
  update_has_limits();
}

void StateAbstract::update_has_limits() { has_limits_ = lb_.allFinite() && ub_.allFinite(); }

}  // namespace crocoddyl
