///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

/* This actuation model is aimed for those robots whose base_link is actuated using a propulsion system, e.g.
 * a multicopter or an aerial manipulator (multicopter with a robotic arm attached).
 * Control input: the thrust (force) created by each propeller.
 * tau_f matrix: this matrix relates the thrust of each propeller to the net force and torque that it causes to the
 * base_link. For a simple quadrotor: tau_f.nrows = 6, tau_f.ncols = 4
 *
 * Reference: M. Geisert and N. Mansard, "Trajectory generation for quadrotor based systems using numerical optimal
 * control," 2016 IEEE International Conference on Robotics and Automation (ICRA), Stockholm, 2016, pp. 2958-2964. See
 * Section III.C  */

namespace crocoddyl {
template <typename _Scalar>
class ActuationModelMultiCopterBaseTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActuationModelMultiCopterBaseTpl(boost::shared_ptr<StateMultibody> state, const std::size_t n_rotors,
                                   const Eigen::Ref<const MatrixXs>& tau_f)
      : Base(state, state->get_nv() - 6 + n_rotors), n_rotors_(n_rotors) {
    pinocchio::JointModelFreeFlyerTpl<Scalar> ff_joint;
    if (state->get_pinocchio()->joints[1].shortname() != ff_joint.shortname()) {
      throw_pretty("Invalid argument: "
                   << "the first joint has to be free-flyer");
    }

    tau_f_ = MatrixXs::Zero(state_->get_nv(), nu_);
    tau_f_.block(0, 0, 6, n_rotors_) = tau_f;
    if (nu_ > n_rotors_) {
      tau_f_.bottomRightCorner(nu_ - n_rotors_, nu_ - n_rotors_) =
          MatrixXs::Identity(nu_ - n_rotors_, nu_ - n_rotors_);
    }
  };
  virtual ~ActuationModelMultiCopterBaseTpl(){};

  virtual void calc(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& /*x*/,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }

    data->tau.noalias() = tau_f_ * u;
  }

  virtual void calcDiff(const boost::shared_ptr<Data>& /*data*/, const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>& /*u*/) {
    // The derivatives has constant values which were set in createData.
  }

  boost::shared_ptr<Data> createData() {
    boost::shared_ptr<Data> data = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    data->dtau_du = tau_f_;
    return data;
  }

  std::size_t get_nrotors() const { return n_rotors_; };
  const MatrixXs& get_tauf() const { return tau_f_; };
  void set_tauf(const Eigen::Ref<const MatrixXs>& tau_f) { tau_f_ = tau_f; }

 protected:
  // Specific of multicopter
  MatrixXs tau_f_;  // Matrix from rotors thrust to body force/moments
  std::size_t n_rotors_;

  using Base::nu_;
  using Base::state_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_
