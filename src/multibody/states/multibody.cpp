///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl {

StateMultibody::StateMultibody(pinocchio::Model& model)
    : StateAbstract(model.nq + model.nv, 2 * model.nv),
      pinocchio_(model),
      x0_(Eigen::VectorXd::Zero(model.nq + model.nv)) {
  x0_.head(nq_) = pinocchio::neutral(pinocchio_);
}

StateMultibody::~StateMultibody() {}

Eigen::VectorXd StateMultibody::zero() const { return x0_; }

Eigen::VectorXd StateMultibody::rand() const {
  Eigen::VectorXd xrand = Eigen::VectorXd::Random(nx_);
  xrand.head(nq_) = pinocchio::randomConfiguration(pinocchio_);
  return xrand;
}

void StateMultibody::diff(const Eigen::Ref<const Eigen::VectorXd>& x0,
                          const Eigen::Ref<const Eigen::VectorXd>& x1,
                          Eigen::Ref<Eigen::VectorXd> dxout) const {
  assert(static_cast<std::size_t>(x0.size()) == nx_ && "x0 has wrong dimension");
  assert(static_cast<std::size_t>(x1.size()) == nx_ && "x1 has wrong dimension");
  assert(static_cast<std::size_t>(dxout.size()) == ndx_ && "Output must be pre-allocated");

  pinocchio::difference(pinocchio_, x0.head(nq_), x1.head(nq_), dxout.head(nv_));
  dxout.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
}

void StateMultibody::integrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                               const Eigen::Ref<const Eigen::VectorXd>& dx,
                               Eigen::Ref<Eigen::VectorXd> xout) const {
  assert(static_cast<std::size_t>(x.size()) == nx_ && "x has wrong dimension");
  assert(static_cast<std::size_t>(dx.size()) == ndx_ && "dx has wrong dimension");
  assert(static_cast<std::size_t>(xout.size()) == nx_ && "Output must be pre-allocated");

  pinocchio::integrate(pinocchio_, x.head(nq_), dx.head(nv_), xout.head(nq_));
  xout.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

void StateMultibody::Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0,
                           const Eigen::Ref<const Eigen::VectorXd>& x1,
                           Eigen::Ref<Eigen::MatrixXd> Jfirst,
                           Eigen::Ref<Eigen::MatrixXd> Jsecond,
                           Jcomponent firstsecond) const {
  assert(static_cast<std::size_t>(x0.size()) == nx_ && "x0 has wrong dimension");
  assert(static_cast<std::size_t>(x1.size()) == nx_ && "x1 has wrong dimension");
  assert(is_a_Jcomponent(firstsecond) && ("firstsecond must be one of the Jcomponent {both, first, second}"));

  typedef Eigen::Block<Eigen::Ref<Eigen::MatrixXd >, -1, 1, true> NColAlignedVectorBlock;
  typedef Eigen::Block<Eigen::Ref<Eigen::MatrixXd> > MatrixBlock;

  if (firstsecond == first) {
    assert(static_cast<std::size_t>(Jfirst.rows()) == ndx_
           && static_cast<std::size_t>(Jfirst.cols()) == ndx_
           && "Jfirst must be of the good size");
    Jfirst.setZero();
    NColAlignedVectorBlock dx_ = Jfirst.rightCols<1>();
    MatrixBlock Jdq_ = Jfirst.bottomLeftCorner(nv_, nv_);

    diff(x1, x0, dx_);
    pinocchio::dIntegrate(pinocchio_, x1.head(nq_), dx_.topRows(nv_),
                          Jdq_ , pinocchio::ARG1);
    updateJdiff(Jdq_, Jfirst.topLeftCorner(nv_, nv_), false);

    Jdq_.setZero();
    dx_.setZero();
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = -Eigen::VectorXd::Ones(nv_);
  } else if (firstsecond == second) {
    assert(static_cast<std::size_t>(Jsecond.rows()) == ndx_
           && static_cast<std::size_t>(Jsecond.cols()) == ndx_
           && "Jsecond must be of the good size");

    Jsecond.setZero();
    NColAlignedVectorBlock dx_ = Jsecond.rightCols<1>();
    MatrixBlock Jdq_ = Jsecond.bottomLeftCorner(nv_, nv_);

    diff(x0, x1, dx_);
    pinocchio::dIntegrate(pinocchio_, x0.head(nq_), dx_.topRows(nv_),
                          Jdq_, pinocchio::ARG1);
    updateJdiff(Jdq_, Jsecond.topLeftCorner(nv_, nv_));

    Jdq_.setZero();
    dx_.setZero();
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else {  // computing both
    assert(static_cast<std::size_t>(Jfirst.rows()) == ndx_ && static_cast<std::size_t>(Jfirst.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    assert(static_cast<std::size_t>(Jsecond.rows()) == ndx_ && static_cast<std::size_t>(Jsecond.cols()) == ndx_ &&
           "Jsecond must be of the good size");
    Jfirst.setZero();
    Jsecond.setZero();

    // Computing Jfirst
    NColAlignedVectorBlock dx1_ = Jfirst.rightCols<1>();
    MatrixBlock Jdq1_ = Jfirst.bottomLeftCorner(nv_, nv_);
    
    diff(x1, x0, dx1_);
    pinocchio::dIntegrate(pinocchio_, x1.head(nq_), dx1_.topRows(nv_),
                          Jdq1_, pinocchio::ARG1);
    updateJdiff(Jdq1_, Jfirst.topLeftCorner(nv_, nv_), false);
    Jdq1_.setZero();
    dx1_.setZero();
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = -Eigen::VectorXd::Ones(nv_);

    // Computing Jsecond
    NColAlignedVectorBlock dx2_ = Jsecond.rightCols<1>();
    MatrixBlock Jdq2_ = Jsecond.bottomLeftCorner(nv_, nv_);
    
    diff(x0, x1, dx2_);
    pinocchio::dIntegrate(pinocchio_, x0.head(nq_), dx2_.topRows(nv_),
                          Jdq2_, pinocchio::ARG1);
    updateJdiff(Jdq2_, Jsecond.topLeftCorner(nv_, nv_));

    dx2_.setZero();
    Jdq2_.setZero();
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  }
}

void StateMultibody::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x,
                                const Eigen::Ref<const Eigen::VectorXd>& dx,
                                Eigen::Ref<Eigen::MatrixXd> Jfirst,
                                Eigen::Ref<Eigen::MatrixXd> Jsecond,
                                Jcomponent firstsecond) const {
  assert(static_cast<std::size_t>(x.size()) == nx_ && "x has wrong dimension");
  assert(static_cast<std::size_t>(dx.size()) == ndx_ && "dx has wrong dimension");
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("firstsecond must be one of the Jcomponent {both, first, second}"));

  if (firstsecond == first) {
    assert(static_cast<std::size_t>(Jfirst.rows()) == ndx_ && static_cast<std::size_t>(Jfirst.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    Jfirst.setZero();

    pinocchio::dIntegrate(pinocchio_, x.head(nq_), dx.head(nv_),
                          Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0);
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else if (firstsecond == second) {
    assert(static_cast<std::size_t>(Jsecond.rows()) == ndx_ && static_cast<std::size_t>(Jsecond.cols()) == ndx_ &&
           "Jsecond must be of the good size");
    Jsecond.setZero();

    pinocchio::dIntegrate(pinocchio_, x.head(nq_), dx.head(nv_),
                          Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1);
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  } else {  // computing both
    assert(static_cast<std::size_t>(Jfirst.rows()) == ndx_ && static_cast<std::size_t>(Jfirst.cols()) == ndx_ &&
           "Jfirst must be of the good size");
    assert(static_cast<std::size_t>(Jsecond.rows()) == ndx_ && static_cast<std::size_t>(Jsecond.cols()) == ndx_ &&
           "Jsecond must be of the good size");

    // Computing Jfirst
    Jfirst.setZero();
    pinocchio::dIntegrate(pinocchio_, x.head(nq_), dx.head(nv_),
                          Jfirst.topLeftCorner(nv_, nv_), pinocchio::ARG0);
    Jfirst.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);

    // Computing Jsecond
    Jsecond.setZero();
    pinocchio::dIntegrate(pinocchio_, x.head(nq_), dx.head(nv_),
                          Jsecond.topLeftCorner(nv_, nv_), pinocchio::ARG1);
    Jsecond.bottomRightCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_);
  }
}

pinocchio::Model& StateMultibody::get_pinocchio() const { return pinocchio_; }

void StateMultibody::updateJdiff(const Eigen::Ref<const Eigen::MatrixXd>& Jdq,
                                 Eigen::Ref<Eigen::MatrixXd> Jd,
                                 bool positive) const {
  if (positive) {
    Jd.diagonal() = Jdq.diagonal();
    Jd.block<6, 6>(0, 0) = Jdq.block<6, 6>(0, 0).inverse();
  } else {
    Jd.diagonal() = -Jdq.diagonal();
    Jd.block<6, 6>(0, 0) = -Jdq.block<6, 6>(0, 0).inverse();
  }
}

}  // namespace crocoddyl
