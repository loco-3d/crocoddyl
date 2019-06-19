#include <crocoddyl/core/states/state-euclidean.hpp>
#include <iostream>
namespace crocoddyl {


StateVector::StateVector(const unsigned int& nx_) : StateAbstract(nx_, nx_) {}

StateVector::~StateVector() {}

Eigen::VectorXd StateVector::zero() {
  return Eigen::VectorXd::Zero(nx_);
}

Eigen::VectorXd StateVector::rand() {
  return Eigen::VectorXd::Random(nx_);
}

void StateVector::diff(const Eigen::Ref<const Eigen::VectorXd>& x0,
                       const Eigen::Ref<const Eigen::VectorXd>& x1,
                       Eigen::Ref<Eigen::VectorXd> dxout) {
  assert(dxout.size() == ndx_ && 
         "StateVector::diff: output must be pre-allocated");
  dxout = x1 - x0;
}

void StateVector::integrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                            const Eigen::Ref<const Eigen::VectorXd>& dx,
                            Eigen::Ref<Eigen::VectorXd> xout) {
  assert(xout.size() == nx_ && 
         "StateVector::diff: output must be pre-allocated");
  xout = x + dx;
}

void StateVector::Jdiff(const Eigen::Ref<const Eigen::VectorXd>&,
                        const Eigen::Ref<const Eigen::VectorXd>&,
                        Eigen::Ref<Eigen::MatrixXd> Jfirst,
                        Eigen::Ref<Eigen::MatrixXd> Jsecond,
                        Jcomponent firstsecond) {
  assert((firstsecond == Jcomponent::first ||
         firstsecond == Jcomponent::second ||
         firstsecond == Jcomponent::both) && (
         "StateVector::Jdiff: firstsecond must be one of the Jcomponent "
         "{both, first, second }"));
  if(firstsecond == Jcomponent::first || firstsecond == Jcomponent::both)
  {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ &&
           "StateVector::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, -1.);
  }
  if(firstsecond == Jcomponent::second || firstsecond == Jcomponent::both)
  {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateVector::Jdiff: Jfirst must be of the good size");
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

void StateVector::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&,
                             const Eigen::Ref<const Eigen::VectorXd>&,
                             Eigen::Ref<Eigen::MatrixXd> Jfirst,
                             Eigen::Ref<Eigen::MatrixXd> Jsecond,
                             Jcomponent firstsecond) {
  assert((firstsecond == Jcomponent::first ||
         firstsecond == Jcomponent::second ||
         firstsecond == Jcomponent::both) && (
         "StateVector::Jdiff: firstsecond must be one of the Jcomponent "
         "{both, first, second }"));
  if(firstsecond == Jcomponent::first || firstsecond == Jcomponent::both)
  {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ &&
           "StateVector::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    Jfirst.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
  if(firstsecond == Jcomponent::second || firstsecond == Jcomponent::both)
  {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ &&
           "StateVector::Jdiff: Jfirst must be of the good size");
    Jsecond.setZero();
    Jsecond.diagonal() = Eigen::VectorXd::Constant(ndx_, 1.);
  }
}

}  // namespace crocoddyl
