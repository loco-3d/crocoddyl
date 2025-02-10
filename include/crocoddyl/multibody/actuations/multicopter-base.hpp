///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, IRI: CSIC-UPC, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_

#include <iostream>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Multicopter actuation model
 *
 * This actuation model is aimed for those robots whose base_link is actuated
 * using a propulsion system, e.g., a multicopter or an aerial manipulator
 * (multicopter with a robotic arm attached). Control input: the thrust (force)
 * created by each propeller. tau_f matrix: this matrix relates the thrust of
 * each propeller to the net force and torque that it causes to the base_link.
 * For a simple quadrotor: tau_f.nrows = 6, tau_f.ncols = 4
 *
 * Both actuation and Jacobians are computed analytically by `calc` and
 * `calcDiff`, respectively.
 *
 * Reference: M. Geisert and N. Mansard, "Trajectory generation for quadrotor
 * based systems using numerical optimal control," 2016 IEEE International
 * Conference on Robotics and Automation (ICRA), Stockholm, 2016, pp. 2958-2964.
 * See Section III.C.
 *
 * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelMultiCopterBaseTpl
    : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  /**
   * @brief Initialize the multicopter actuation model
   *
   * @param[in] state  State of the dynamical system
   * @param[in] tau_f  Matrix that maps the thrust of each propeller to the net
   * force and torque
   */
  DEPRECATED("Use constructor ActuationModelFloatingBaseThrustersTpl",
             ActuationModelMultiCopterBaseTpl(
                 std::shared_ptr<StateMultibody> state,
                 const Eigen::Ref<const Matrix6xs>& tau_f));

  DEPRECATED("Use constructor without n_rotors",
             ActuationModelMultiCopterBaseTpl(
                 std::shared_ptr<StateMultibody> state,
                 const std::size_t n_rotors,
                 const Eigen::Ref<const Matrix6xs>& tau_f));
  virtual ~ActuationModelMultiCopterBaseTpl() {}

  virtual void calc(const std::shared_ptr<Data>& data,
                    const Eigen::Ref<const VectorXs>&,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }

    data->tau.noalias() = tau_f_ * u;
  }

#ifndef NDEBUG
  virtual void calcDiff(const std::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) {
#else
  virtual void calcDiff(const std::shared_ptr<Data>&,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) {
#endif
    // The derivatives has constant values which were set in createData.
    assert_pretty(MatrixXs(data->dtau_du).isApprox(tau_f_),
                  "dtau_du has wrong value");
  }

  virtual void commands(const std::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>& tau) {
    data->u.noalias() = Mtau_ * tau;
  }

#ifndef NDEBUG
  virtual void torqueTransform(const std::shared_ptr<Data>& data,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#else
  virtual void torqueTransform(const std::shared_ptr<Data>&,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#endif
    // The torque transform has constant values which were set in createData.
    assert_pretty(MatrixXs(data->Mtau).isApprox(Mtau_), "Mtau has wrong value");
  }

  std::shared_ptr<Data> createData() {
    std::shared_ptr<Data> data =
        std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    data->dtau_du = tau_f_;
    data->Mtau = Mtau_;
    for (std::size_t i = 0; i < 2; ++i) {
      data->tau_set[i] = false;
    }
    return data;
  }

  std::size_t get_nrotors() const { return n_rotors_; };
  const MatrixXs& get_tauf() const { return tau_f_; };
  void set_tauf(const Eigen::Ref<const MatrixXs>& tau_f) { tau_f_ = tau_f; }

 protected:
  MatrixXs tau_f_;  //!< Matrix from rotors thrust to body force/moments
  MatrixXs Mtau_;   //!< Constaint torque transform from generalized torques to
                    //!< joint torque inputs
  std::size_t n_rotors_;  //!< Number of rotors

  using Base::nu_;
  using Base::state_;

#ifndef NDEBUG
 private:
  MatrixXs S_;
#endif
};

template <typename Scalar>
ActuationModelMultiCopterBaseTpl<Scalar>::ActuationModelMultiCopterBaseTpl(
    std::shared_ptr<StateMultibody> state,
    const Eigen::Ref<const Matrix6xs>& tau_f)
    : Base(state, state->get_nv() - 6 + tau_f.cols()), n_rotors_(tau_f.cols()) {
  pinocchio::JointModelFreeFlyerTpl<Scalar> ff_joint;
  if (state->get_pinocchio()->joints[1].shortname() != ff_joint.shortname()) {
    throw_pretty(
        "Invalid argument: " << "the first joint has to be free-flyer");
  }

  tau_f_ = MatrixXs::Zero(state_->get_nv(), nu_);
  tau_f_.block(0, 0, 6, n_rotors_) = tau_f;
  if (nu_ > n_rotors_) {
    tau_f_.bottomRightCorner(nu_ - n_rotors_, nu_ - n_rotors_)
        .diagonal()
        .setOnes();
  }
  Mtau_ = pseudoInverse(MatrixXs(tau_f));
  std::cerr << "Deprecated ActuationModelMultiCopterBase: Use "
               "ActuationModelFloatingBaseThrusters"
            << std::endl;
}

template <typename Scalar>
ActuationModelMultiCopterBaseTpl<Scalar>::ActuationModelMultiCopterBaseTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t n_rotors,
    const Eigen::Ref<const Matrix6xs>& tau_f)
    : Base(state, state->get_nv() - 6 + n_rotors), n_rotors_(n_rotors) {
  pinocchio::JointModelFreeFlyerTpl<Scalar> ff_joint;
  if (state->get_pinocchio()->joints[1].shortname() != ff_joint.shortname()) {
    throw_pretty(
        "Invalid argument: " << "the first joint has to be free-flyer");
  }

  tau_f_ = MatrixXs::Zero(state_->get_nv(), nu_);
  tau_f_.block(0, 0, 6, n_rotors_) = tau_f;
  if (nu_ > n_rotors_) {
    tau_f_.bottomRightCorner(nu_ - n_rotors_, nu_ - n_rotors_)
        .diagonal()
        .setOnes();
  }
  std::cerr << "Deprecated ActuationModelMultiCopterBase: Use constructor "
               "without n_rotors."
            << std::endl;
}

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_
