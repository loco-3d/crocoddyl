///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"

namespace crocoddyl {

DifferentialActionModelLQR::DifferentialActionModelLQR(const std::size_t& nq, const std::size_t& nu, bool drift_free)
    : DifferentialActionModelAbstract(boost::make_shared<StateVector>(2 * nq), nu), drift_free_(drift_free) {
  // TODO(cmastalli): substitute by random (vectors) and random-orthogonal (matrices)
  Fq_ = Eigen::MatrixXd::Identity(state_->get_nq(), state_->get_nq());
  Fv_ = Eigen::MatrixXd::Identity(state_->get_nv(), state_->get_nv());
  Fu_ = Eigen::MatrixXd::Identity(state_->get_nq(), nu_);
  f0_ = Eigen::VectorXd::Ones(state_->get_nv());
  Lxx_ = Eigen::MatrixXd::Identity(state_->get_nx(), state_->get_nx());
  Lxu_ = Eigen::MatrixXd::Identity(state_->get_nx(), nu_);
  Luu_ = Eigen::MatrixXd::Identity(nu_, nu_);
  lx_ = Eigen::VectorXd::Ones(state_->get_nx());
  lu_ = Eigen::VectorXd::Ones(nu_);
}

DifferentialActionModelLQR::~DifferentialActionModelLQR() {}

void DifferentialActionModelLQR::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                      const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  if (drift_free_) {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u;
  } else {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u + f0_;
  }
  data->cost = 0.5 * x.dot(Lxx_ * x) + 0.5 * u.dot(Luu_ * u) + x.dot(Lxu_ * u) + lx_.dot(x) + lu_.dot(u);
}

void DifferentialActionModelLQR::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
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
  data->Fx.leftCols(state_->get_nq()) = Fq_;
  data->Fx.rightCols(state_->get_nv()) = Fv_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelLQR::createData() {
  return boost::make_shared<DifferentialActionDataLQR>(this);
}

const Eigen::MatrixXd& DifferentialActionModelLQR::get_Fq() const { return Fq_; }

const Eigen::MatrixXd& DifferentialActionModelLQR::get_Fv() const { return Fv_; }

const Eigen::MatrixXd& DifferentialActionModelLQR::get_Fu() const { return Fu_; }

const Eigen::VectorXd& DifferentialActionModelLQR::get_f0() const { return f0_; }

const Eigen::VectorXd& DifferentialActionModelLQR::get_lx() const { return lx_; }

const Eigen::VectorXd& DifferentialActionModelLQR::get_lu() const { return lu_; }

const Eigen::MatrixXd& DifferentialActionModelLQR::get_Lxx() const { return Lxx_; }

const Eigen::MatrixXd& DifferentialActionModelLQR::get_Lxu() const { return Lxu_; }

const Eigen::MatrixXd& DifferentialActionModelLQR::get_Luu() const { return Luu_; }

void DifferentialActionModelLQR::set_Fq(const Eigen::MatrixXd& Fq) {
  if (static_cast<std::size_t>(Fq.rows()) != state_->get_nq() ||
      static_cast<std::size_t>(Fq.cols()) != state_->get_nq()) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " + std::to_string(state_->get_nq()) + "," +
                        std::to_string(state_->get_nq()) + ")");
  }
  Fq_ = Fq;
}

void DifferentialActionModelLQR::set_Fv(const Eigen::MatrixXd& Fv) {
  if (static_cast<std::size_t>(Fv.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(Fv.cols()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "Fv has wrong dimension (it should be " + std::to_string(state_->get_nv()) + "," +
                        std::to_string(state_->get_nv()) + ")");
  }
  Fv_ = Fv;
}

void DifferentialActionModelLQR::set_Fu(const Eigen::MatrixXd& Fu) {
  if (static_cast<std::size_t>(Fu.rows()) != state_->get_nq() || static_cast<std::size_t>(Fu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " + std::to_string(state_->get_nq()) + "," +
                        std::to_string(nu_) + ")");
  }
  Fu_ = Fu;
}

void DifferentialActionModelLQR::set_f0(const Eigen::VectorXd& f0) {
  if (static_cast<std::size_t>(f0.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "f0 has wrong dimension (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  f0_ = f0;
}

void DifferentialActionModelLQR::set_lx(const Eigen::VectorXd& lx) {
  if (static_cast<std::size_t>(lx.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "lx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  lx_ = lx;
}

void DifferentialActionModelLQR::set_lu(const Eigen::VectorXd& lu) {
  if (static_cast<std::size_t>(lu.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lu has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  lu_ = lu;
}

void DifferentialActionModelLQR::set_Lxx(const Eigen::MatrixXd& Lxx) {
  if (static_cast<std::size_t>(Lxx.rows()) != state_->get_nx() ||
      static_cast<std::size_t>(Lxx.cols()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "Lxx has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(state_->get_nx()) + ")");
  }
  Lxx_ = Lxx;
}

void DifferentialActionModelLQR::set_Lxu(const Eigen::MatrixXd& Lxu) {
  if (static_cast<std::size_t>(Lxu.rows()) != state_->get_nx() || static_cast<std::size_t>(Lxu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Lxu has wrong dimension (it should be " + std::to_string(state_->get_nx()) + "," +
                        std::to_string(nu_) + ")");
  }
  Lxu_ = Lxu;
}

void DifferentialActionModelLQR::set_Luu(const Eigen::MatrixXd& Luu) {
  if (static_cast<std::size_t>(Luu.rows()) != nu_ || static_cast<std::size_t>(Luu.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "Fq has wrong dimension (it should be " + std::to_string(nu_) + "," + std::to_string(nu_) + ")");
  }
  Luu_ = Luu;
}

}  // namespace crocoddyl
