#include <crocoddyl/core/states/state-euclidean.hpp>

namespace crocoddyl {


StateVector::StateVector(const unsigned int& nx) : StateAbstract(nx, nx) {}

StateVector::~StateVector() {}

Eigen::VectorXd StateVector::zero() {
  return Eigen::VectorXd::Zero(nx);
}

Eigen::VectorXd StateVector::rand() {
  return Eigen::VectorXd::Random(nx);
}

void StateVector::diff(const Eigen::Ref<const Eigen::VectorXd>& x0,
                       const Eigen::Ref<const Eigen::VectorXd>& x1,
                       Eigen::Ref<Eigen::VectorXd> dxout) {
  dxout = x1 - x0;
}

void StateVector::integrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                            const Eigen::Ref<const Eigen::VectorXd>& dx,
                            Eigen::Ref<Eigen::VectorXd> xout) {
  xout = x + dx;
}

void StateVector::Jdiff(const Eigen::Ref<const Eigen::VectorXd>&,
                        const Eigen::Ref<const Eigen::VectorXd>&,
                        Eigen::Ref<Eigen::MatrixXd> Jfirst,
                        Eigen::Ref<Eigen::MatrixXd> Jsecond,
                       Jcomponent firstsecond) {
  switch (firstsecond) {
  case first: {
    Jfirst = Eigen::MatrixXd::Zero(ndx, ndx);
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx, -1.);
    break;
  } case second: {
    Jsecond.setIdentity(ndx, ndx);
    break;
  } case both: {
    Jfirst = Eigen::MatrixXd::Zero(ndx, ndx);
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx, -1.);
    Jsecond.setIdentity(ndx, ndx);
    break;
  } default: {
    Jfirst = Eigen::MatrixXd::Zero(ndx, ndx);
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx, -1.);
    Jsecond.setIdentity(ndx, ndx);
  }}
}

void StateVector::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&,
                             const Eigen::Ref<const Eigen::VectorXd>&,
                             Eigen::Ref<Eigen::MatrixXd> Jfirst,
                             Eigen::Ref<Eigen::MatrixXd> Jsecond,
                             Jcomponent firstsecond) {
  switch (firstsecond) {
  case first: {
    Jfirst.setIdentity(ndx, ndx);
    break;
  } case second: {
    Jsecond.setIdentity(ndx, ndx);
    break;
  } case both: {
    Jfirst.setIdentity(ndx, ndx);
    Jsecond.setIdentity(ndx, ndx);
    break;
  } default: {
    Jfirst.setIdentity(ndx, ndx);
    Jfirst *= -1.;
    Jsecond.setIdentity(ndx, ndx);
  }}
}

}  // namespace crocoddyl
