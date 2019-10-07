///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

ActionModelAbstract::ActionModelAbstract(StateAbstract& state, const std::size_t& nu, const std::size_t& nr)
    : nu_(nu),
      nr_(nr),
      state_(state),
      unone_(Eigen::VectorXd::Zero(nu)),
      u_lb_(Eigen::VectorXd::Constant(nu, -std::numeric_limits<double>::infinity())),
      u_ub_(Eigen::VectorXd::Constant(nu, std::numeric_limits<double>::infinity())),
      has_control_limits_(false) {}

ActionModelAbstract::~ActionModelAbstract() {}

void ActionModelAbstract::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                               const Eigen::Ref<const Eigen::VectorXd>& x) {
  calc(data, x, unone_);
}

void ActionModelAbstract::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>& x) {
  calcDiff(data, x, unone_);
}

void ActionModelAbstract::quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                                      const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t& maxiter,
                                      const double& tol) {
  assert((u.size() == nu_ || nu_ == 0) && "u has wrong dimension");
  assert(x.size() == state_.get_nx() && "x has wrong dimension");

  const std::size_t& ndx = state_.get_ndx();
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndx);
  if (nu_ == 0) {
    // TODO(cmastalli): create a method for autonomous systems
  } else {
    Eigen::VectorXd du = Eigen::VectorXd::Zero(nu_);
    for (std::size_t i = 0; i < maxiter; ++i) {
      calcDiff(data, x, u);
      state_.diff(x, data->xnext, dx);
      du = -pseudoInverse(data->Fu) * data->Fx * dx;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}

boost::shared_ptr<ActionDataAbstract> ActionModelAbstract::createData() {
  return boost::make_shared<ActionDataAbstract>(this);
}

const std::size_t& ActionModelAbstract::get_nu() const { return nu_; }

const std::size_t& ActionModelAbstract::get_nr() const { return nr_; }

StateAbstract& ActionModelAbstract::get_state() const { return state_; }

const Eigen::VectorXd& ActionModelAbstract::get_u_lb() const { return u_lb_; }

const Eigen::VectorXd& ActionModelAbstract::get_u_ub() const { return u_ub_; }

void ActionModelAbstract::set_u_lb(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
  assert(nu_ == u_in.size() && "Number of rows of u_in must match nu_");
  u_lb_ = u_in;
  update_has_control_limits();
}

void ActionModelAbstract::set_u_ub(const Eigen::Ref<const Eigen::VectorXd>& u_in) {
  assert(nu_ == u_in.size() && "Number of rows of u_in must match nu_");
  u_ub_ = u_in;
  update_has_control_limits();
}

bool const& ActionModelAbstract::get_has_control_limits() const { return has_control_limits_; }

void ActionModelAbstract::update_has_control_limits() { has_control_limits_ = u_lb_.allFinite() && u_ub_.allFinite(); }

}  // namespace crocoddyl
