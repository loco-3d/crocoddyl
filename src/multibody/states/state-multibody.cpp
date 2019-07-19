#include "crocoddyl/multibody/states/state-multibody.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

namespace crocoddyl {

StateMultibody::StateMultibody(pinocchio::Model& model)
    : StateAbstract(model.nq + model.nv, 2 * model.nv),
      model_(model),
      x0_(Eigen::VectorXd::Zero(model.nq + model.nv)),
      dx_(Eigen::VectorXd::Zero(2 * model.nv)),
      Jdq_(Eigen::MatrixXd::Zero(model.nv, model.nv)) {
  x0_.head(nq_) = pinocchio::neutral(model_);
}

StateMultibody::~StateMultibody() {}

Eigen::VectorXd StateMultibody::zero() { return x0_; }

Eigen::VectorXd StateMultibody::rand() {
  Eigen::VectorXd xrand = Eigen::VectorXd::Random(nx_);
  xrand.head(nq_) = pinocchio::randomConfiguration(model_);
  return xrand;
}

void StateMultibody::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                          Eigen::Ref<Eigen::VectorXd> dxout) {
  assert(x0.size() == nx_ && "StateMultibody::diff: x0 has wrong dimension");
  assert(x1.size() == nx_ && "StateMultibody::diff: x1 has wrong dimension");
  assert(dxout.size() == ndx_ && "StateMultibody::diff: output must be pre-allocated");

  const Eigen::VectorXd& q0 = x0.head(nq_);
  const Eigen::VectorXd& v0 = x0.tail(nv_);
  const Eigen::VectorXd& q1 = x1.head(nq_);
  const Eigen::VectorXd& v1 = x1.tail(nv_);
  dxout << pinocchio::difference(model_, q0, q1), v1 - v0;
}

void StateMultibody::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                               Eigen::Ref<Eigen::VectorXd> xout) {
  assert(x.size() == nx_ && "StateMultibody::diff: x has wrong dimension");
  assert(dx.size() == ndx_ && "StateMultibody::diff: dx has wrong dimension");
  assert(xout.size() == nx_ && "StateMultibody::diff: output must be pre-allocated");

  const Eigen::VectorXd& q = x.head(nq_);
  const Eigen::VectorXd& v = x.tail(nv_);
  const Eigen::VectorXd& dq = dx.head(nq_);
  const Eigen::VectorXd& dv = dx.tail(nv_);
  xout << pinocchio::integrate(model_, q, dq), v - dv;
}

void StateMultibody::Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                           Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                           Jcomponent firstsecond) {
  assert(x0.size() == nx_ && "StateMultibody::Jdiff: x0 has wrong dimension");
  assert(x1.size() == nx_ && "StateMultibody::Jdiff: x1 has wrong dimension");
  assert((firstsecond == Jcomponent::first || firstsecond == Jcomponent::second || firstsecond == Jcomponent::both) &&
         ("StateMultibody::Jdiff: firstsecond must be one of the Jcomponent "
          "{both, first, second}"));

  if (firstsecond == first) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateMultibody::Jdiff: Jfirst must be of the good size");

    diff(x1, x0, dx_);
    const Eigen::VectorXd& q1 = x1.head(nq_);
    const Eigen::VectorXd& dq1 = dx_.head(nv_);
    pinocchio::dIntegrate(model_, q1, dq1, Jdq_, pinocchio::ARG1);

    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = -Jdq_.inverse();
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = -Eigen::VectorXd::Ones(nv_);
  } else if (firstsecond == second) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    diff(x0, x1, dx_);
    const Eigen::VectorXd& q0 = x0.head(nq_);
    const Eigen::VectorXd& dq0 = dx_.head(nv_);
    pinocchio::dIntegrate(model_, q0, dq0, Jdq_, pinocchio::ARG1);

    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Jdq_.inverse();
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else {  // computing both
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateMultibody::Jdiff: Jfirst must be of the good size");
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    // Computing Jfirst
    diff(x1, x0, dx_);
    const Eigen::VectorXd& q1 = x1.head(nq_);
    const Eigen::VectorXd& dq1 = dx_.head(nv_);
    pinocchio::dIntegrate(model_, q1, dq1, Jdq_, pinocchio::ARG1);
    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = -Jdq_.inverse();
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = -Eigen::VectorXd::Ones(nv_);

    // Computing Jsecond
    diff(x0, x1, dx_);
    const Eigen::VectorXd& q0 = x0.head(nq_);
    const Eigen::VectorXd& dq0 = dx_.head(nv_);
    pinocchio::dIntegrate(model_, q0, dq0, Jdq_, pinocchio::ARG1);
    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Jdq_.inverse();
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  }
}

void StateMultibody::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                                const Eigen::Ref<const Eigen::VectorXd>& dx, Eigen::Ref<Eigen::MatrixXd> Jfirst,
                                Eigen::Ref<Eigen::MatrixXd> Jsecond, Jcomponent firstsecond) {
  assert(x.size() == nx_ && "StateMultibody::Jintegrate: x has wrong dimension");
  assert(dx.size() == ndx_ && "StateMultibody::Jintegrate: dx has wrong dimension");
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("StateMultibody::Jintegrate: firstsecond must be one of the Jcomponent "
          "{both, first, second}"));

  const Eigen::VectorXd& q = x.head(nq_);
  const Eigen::VectorXd& dq = dx.head(nq_);
  if (firstsecond == first) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ &&
           "StateMultibody::Jintegrate: Jfirst must be of the good size");

    pinocchio::dIntegrate(model_, q, dq, Jdq_, pinocchio::ARG0);
    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = Jdq_;
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else if (firstsecond == second) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    pinocchio::dIntegrate(model_, q, dq, Jdq_, pinocchio::ARG1);
    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Jdq_;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else {  // computing both
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateMultibody::Jdiff: Jfirst must be of the good size");
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateMultibody::Jdiff: Jsecond must be of the good size");

    // Computing Jfirst
    pinocchio::dIntegrate(model_, q, dq, Jdq_, pinocchio::ARG0);
    Jfirst.setZero();
    Jfirst.topLeftCorner(nv_, nv_) = Jdq_;
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);

    // Computing Jsecond
    pinocchio::dIntegrate(model_, q, dq, Jdq_, pinocchio::ARG1);
    Jsecond.setZero();
    Jsecond.topLeftCorner(nv_, nv_) = Jdq_;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  }
}

}  // namespace crocoddyl
