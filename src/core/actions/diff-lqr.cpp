#include "crocoddyl/core/actions/diff-lqr.hpp"

namespace crocoddyl {

DifferentialActionModelLQR::DifferentialActionModelLQR(const unsigned int& nq, const unsigned int& nu, bool drift_free)
    : DifferentialActionModelAbstract(new StateVector(2 * nq), nu), drift_free_(drift_free) {
  // TODO substitute by random (vectors) and random-orthogonal (matrices)
  Fq_ = Eigen::MatrixXd::Identity(nq_, nq_);
  Fv_ = Eigen::MatrixXd::Identity(nv_, nv_);
  Fu_ = Eigen::MatrixXd::Identity(nq_, nu_);
  f0_ = Eigen::VectorXd::Ones(nv_);
  Lxx_ = Eigen::MatrixXd::Identity(nx_, nx_);
  Lxu_ = Eigen::MatrixXd::Identity(nx_, nu_);
  Luu_ = Eigen::MatrixXd::Identity(nu_, nu_);
  lx_ = Eigen::VectorXd::Ones(nx_);
  lu_ = Eigen::VectorXd::Ones(nu_);
}

DifferentialActionModelLQR::~DifferentialActionModelLQR() {}

void DifferentialActionModelLQR::calc(boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                      const Eigen::Ref<const Eigen::VectorXd>& u) {
  const Eigen::VectorXd& q = x.head(nq_);
  const Eigen::VectorXd& v = x.tail(nv_);
  if (drift_free_) {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u;
  } else {
    data->xout = Fq_ * q + Fv_ * v + Fu_ * u + f0_;
  }
  data->cost = 0.5 * x.dot(Lxx_ * x) + 0.5 * u.dot(Luu_ * u) + x.dot(Lxu_ * u) + lx_.dot(x) + lu_.dot(u);
}

void DifferentialActionModelLQR::calcDiff(boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                          const Eigen::Ref<const Eigen::VectorXd>& x,
                                          const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  data->Lx = lx_ + Lxx_ * x + Lxu_ * u;
  data->Lu = lu_ + Lxu_.transpose() * x + Luu_ * u;
  data->Fx.leftCols(nq_) = Fq_;
  data->Fx.rightCols(nv_) = Fv_;
  data->Fu = Fu_;
  data->Lxx = Lxx_;
  data->Lxu = Lxu_;
  data->Luu = Luu_;
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelLQR::createData() {
  return boost::make_shared<DifferentialActionDataLQR>(this);
}

}  // namespace crocoddyl