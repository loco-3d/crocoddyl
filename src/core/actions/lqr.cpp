///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actions/lqr.hpp"

namespace crocoddyl {

ActionModelLQR::ActionModelLQR(const std::size_t& nx, const std::size_t& nu, bool drift_free)
    : ActionModelAbstract(boost::make_shared<StateVector>(nx), nu, 0), drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal (matrices)
  Fx_ = Eigen::MatrixXd::Identity(nx, nx);
  Fu_ = Eigen::MatrixXd::Identity(nx, nu);
  f0_ = Eigen::VectorXd::Ones(nx);
  Lxx_ = Eigen::MatrixXd::Identity(nx, nx);
  Lxu_ = Eigen::MatrixXd::Identity(nx, nu);
  Luu_ = Eigen::MatrixXd::Identity(nu, nu);
  lx_ = Eigen::VectorXd::Ones(nx);
  lu_ = Eigen::VectorXd::Ones(nu);
}

ActionModelLQR::~ActionModelLQR() {}

void ActionModelLQR::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  if (drift_free_) {
    data->xnext = Fx_ * x + Fu_ * u;
  } else {
    data->xnext = Fx_ * x + Fu_ * u + f0_;
  }
  data->cost = 0.5 * x.dot(Lxx_ * x) + 0.5 * u.dot(Luu_ * u) + x.dot(Lxu_ * u) + lx_.dot(x) + lu_.dot(u);
}

void ActionModelLQR::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                              const bool& recalc) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  if (recalc) {
    calc(data, x, u);
  }
  data->Lx = lx_ + Lxx_ * x + Lxu_ * u;
  data->Lu = lu_ + Lxu_.transpose() * x + Luu_ * u;
  data->Fx = Fx_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

boost::shared_ptr<ActionDataAbstract> ActionModelLQR::createData() { return boost::make_shared<ActionDataLQR>(this); }

const Eigen::MatrixXd& ActionModelLQR::get_Fx() const { return Fx_; }

const Eigen::MatrixXd& ActionModelLQR::get_Fu() const { return Fu_; }

const Eigen::VectorXd& ActionModelLQR::get_f0() const { return f0_; }

const Eigen::VectorXd& ActionModelLQR::get_lx() const { return lx_; }

const Eigen::VectorXd& ActionModelLQR::get_lu() const { return lu_; }

const Eigen::MatrixXd& ActionModelLQR::get_Lxx() const { return Lxx_; }

const Eigen::MatrixXd& ActionModelLQR::get_Lxu() const { return Lxu_; }

const Eigen::MatrixXd& ActionModelLQR::get_Luu() const { return Luu_; }

void ActionModelLQR::set_Fx(const Eigen::MatrixXd& Fx) {
  if (static_cast<std::size_t>(Fx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Fx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Fx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Fx_ = Fx;
}

void ActionModelLQR::set_Fu(const Eigen::MatrixXd& Fu) {
  if (static_cast<std::size_t>(Fu.rows()) != state_->get_nx() || static_cast<std::size_t>(Fu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  Fu_ = Fu;
}

void ActionModelLQR::set_f0(const Eigen::VectorXd& f0) {
  if (static_cast<std::size_t>(f0.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "f0 has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  f0_ = f0;
}

void ActionModelLQR::set_lx(const Eigen::VectorXd& lx) {
  if (static_cast<std::size_t>(lx.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "lx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  lx_ = lx;
}

void ActionModelLQR::set_lu(const Eigen::VectorXd& lu) {
  if (static_cast<std::size_t>(lu.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lu has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  lu_ = lu;
}

void ActionModelLQR::set_Lxx(const Eigen::MatrixXd& Lxx) {
  if (static_cast<std::size_t>(Lxx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Lxx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Lxx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Lxx_ = Lxx;
}

void ActionModelLQR::set_Lxu(const Eigen::MatrixXd& Lxu) {
  if (static_cast<std::size_t>(Lxu.rows()) != state_->get_nx() || static_cast<std::size_t>(Lxu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Lxu has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  Lxu_ = Lxu;
}

void ActionModelLQR::set_Luu(const Eigen::MatrixXd& Luu) {
  if (static_cast<std::size_t>(Luu.rows()) != nu_ || static_cast<std::size_t>(Luu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  Luu_ = Luu;
}

}  // namespace crocoddyl
