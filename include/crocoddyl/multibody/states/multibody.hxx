///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl {

template <typename Scalar>
StateMultibodyTpl<Scalar>::StateMultibodyTpl(boost::shared_ptr<pinocchio::ModelTpl<Scalar> > model)
    : Base(model->nq + model->nv, 2 * model->nv),
      pinocchio_(model),
      x0_(VectorXs::Zero(model->nq + model->nv)) {
  x0_.head(nq_) = pinocchio::neutral(*pinocchio_.get());

  // In a multibody system, we could define the first joint using Lie groups.
  // The current cases are free-flyer (SE3) and spherical (S03).
  // Instead simple represents any joint that can model within the Euclidean manifold.
  // The rest of joints use Euclidean algebra. We use this fact for computing Jdiff.

  // Define internally the limits of the first joint

  const std::size_t nq0 = model->joints[1].nq();

  lb_.template head(nq0) = -std::numeric_limits<Scalar>::infinity() * VectorXs::Ones(nq0);
  ub_.template head(nq0) = std::numeric_limits<Scalar>::infinity() * VectorXs::Ones(nq0);
  lb_.segment(nq0, nq_ - nq0) = pinocchio_->lowerPositionLimit.tail(nq_ - nq0);
  ub_.segment(nq0, nq_ - nq0) = pinocchio_->upperPositionLimit.tail(nq_ - nq0);
  lb_.tail(nv_) = -pinocchio_->velocityLimit;
  ub_.tail(nv_) = pinocchio_->velocityLimit;
  Base::update_has_limits();
}

template <typename Scalar>
StateMultibodyTpl<Scalar>::~StateMultibodyTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateMultibodyTpl<Scalar>::zero() const {
  return x0_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateMultibodyTpl<Scalar>::rand() const {
  VectorXs xrand = VectorXs::Random(nx_);
  xrand.head(nq_) = pinocchio::randomConfiguration(*pinocchio_.get());
  return xrand;
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                     Eigen::Ref<VectorXs> dxout) const {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }

  pinocchio::difference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), dxout.head(nv_));
  dxout.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                          Eigen::Ref<VectorXs> xout) const {
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "xout has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }

  pinocchio::integrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), xout.head(nq_));
  xout.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                      Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                      const Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }

  if (firstsecond == first) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }

    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                           pinocchio::ARG0);
    Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)-1;
  } else if (firstsecond == second) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                           pinocchio::ARG1);
    Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
  } else {  // computing both
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                           pinocchio::ARG0);
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                           pinocchio::ARG1);
    Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)-1;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
  }
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                           Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                           const Jcomponent firstsecond, const AssignmentOp op) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::SETTO);
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::ADDTO);
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::RMTO);
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::SETTO);
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::ADDTO);
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::RMTO);
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
}

template <typename Scalar>
void StateMultibodyTpl<Scalar>::JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                                                    const Eigen::Ref<const VectorXs>& dx, Eigen::Ref<MatrixXs> Jin,
                                                    const Jcomponent firstsecond) const {
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));

  switch (firstsecond) {
    case first:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jin.topRows(nv_), pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jin.topRows(nv_), pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. both not supported for this operation.");
      break;
  }
}

template <typename Scalar>
const boost::shared_ptr<pinocchio::ModelTpl<Scalar> >& StateMultibodyTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

}  // namespace crocoddyl
