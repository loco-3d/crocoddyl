///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2014-2024, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_PROPELLERS_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_PROPELLERS_HPP_

#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <vector>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

enum PropellerType { CW = 0, CCW };

template <typename _Scalar>
struct PropellerTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef PropellerTpl<Scalar> Propeller;

  /**
   * @brief Initialize the propeller in a give pose from the root joint.
   *
   * @param pose[in]     Pose from root joint
   * @param cthrust[in]  Coefficient of thrust (it relates propeller's (square)
   * velocity to its thrust)
   * @param ctorque[in]  Coefficient of torque (it relates propeller's (square)
   * velocity to its torque)
   * @param type[in]     Type of propeller (clockwise or counterclockwise,
   * default clockwise)
   */
  PropellerTpl(const SE3& pose, const Scalar cthrust, const Scalar ctorque,
               const PropellerType type = CW)
      : pose(pose), cthrust(cthrust), ctorque(ctorque), type(type) {}

  /**
   * @brief Initialize the propeller in a pose in the origin of the root joint.
   *
   * @param cthrust[in]  Coefficient of thrust (it relates propeller's (square)
   * velocity to its thrust)
   * @param ctorque[in]  Coefficient of torque (it relates propeller's (square)
   * velocity to its torque)
   * @param type[in]     Type of propeller (clockwise or counterclockwise,
   * default clockwise)
   */
  PropellerTpl(const Scalar cthrust, const Scalar ctorque,
               const PropellerType type = CW)
      : pose(SE3::Identity()), cthrust(cthrust), ctorque(ctorque), type(type) {}
  PropellerTpl(const PropellerTpl<Scalar>& clone)
      : pose(clone.pose),
        cthrust(clone.cthrust),
        ctorque(clone.ctorque),
        type(clone.type) {}

  PropellerTpl& operator=(const PropellerTpl<Scalar>& other) {
    if (this != &other) {
      pose = other.pose;
      cthrust = other.cthrust;
      ctorque = other.ctorque;
      type = other.type;
    }
    return *this;
  }

  template <typename OtherScalar>
  bool operator==(const PropellerTpl<OtherScalar>& other) const {
    return (pose == other.pose && cthrust == other.cthrust &&
            ctorque == other.ctorque && type == other.type);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const PropellerTpl<Scalar>& X) {
    os << "   pose:" << std::endl
       << X.pose << "cthrust: " << X.cthrust << std::endl
       << "ctorque: " << X.ctorque << std::endl
       << "   type: " << X.type << std::endl;
    return os;
  }

  SE3 pose;            //!< Propeller pose
  Scalar cthrust;      //!< Coefficient of thrust (it relates the square of the
                       //!< propeller velocity with its thrust)
  Scalar ctorque;      //!< Coefficient of torque (it relates the square of the
                       //!< propeller velocity with its torque)
  PropellerType type;  //!< Type of propeller (CW and CCW for clockwise and
                       //!< counterclockwise, respectively)
};

/**
 * @brief Actuation models for floating base systems actuated with propellers
 *
 * This actuation model models floating base robots equipped with propellers,
 * e.g., multicopters or marine robots equipped with manipulators. It control
 * inputs are the propellers' thrust (i.e., forces) and joint torques.
 *
 * Both actuation and Jacobians are computed analytically by `calc` and
 * `calcDiff`, respectively.
 *
 * We assume the robot velocity to zero for easily related square propeller
 * velocities with thrust and torque generated. This approach is similarly
 * implemented in M. Geisert and N. Mansard, "Trajectory generation for
 * quadrotor based systems using numerical optimal control", (ICRA). See Section
 * III.C.
 *
 * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelFloatingBasePropellersTpl
    : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef PropellerTpl<Scalar> Propeller;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the floating base actuation model equipped with
   * propellers
   *
   * @param[in] state       State of the dynamical system
   * @param[in] propellers  Vector of propellers
   */
  ActuationModelFloatingBasePropellersTpl(
      boost::shared_ptr<StateMultibody> state,
      const std::vector<Propeller>& propellers)
      : Base(state,
             state->get_nv() -
                 state->get_pinocchio()
                     ->joints[(
                         state->get_pinocchio()->existJointName("root_joint")
                             ? state->get_pinocchio()->getJointId("root_joint")
                             : 0)]
                     .nv() +
                 propellers.size()),
        propellers_(propellers),
        n_propellers_(propellers.size()),
        W_thrust_(state_->get_nv(), nu_),
        update_data_(true) {
    if (!state->get_pinocchio()->existJointName("root_joint")) {
      throw_pretty(
          "Invalid argument: "
          << "the first joint has to be a root one (e.g., free-flyer joint)");
    }
    // Update the joint actuation part
    W_thrust_.setZero();
    if (nu_ > n_propellers_) {
      W_thrust_
          .template bottomRightCorner(nu_ - n_propellers_, nu_ - n_propellers_)
          .diagonal()
          .setOnes();
    }
    // Update the floating base actuation part
    set_propellers(propellers_);
  }
  virtual ~ActuationModelFloatingBasePropellersTpl() {}

  /**
   * @brief Compute the actuation signal and actuation set from its thrust
   * and joint torque inputs \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   *
   * @param[in] data  Floating base propellers actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<Data>& data,
                    const Eigen::Ref<const VectorXs>&,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu_) + ")");
    }
    if (update_data_) {
      updateData(data);
    }
    data->tau.noalias() = data->dtau_du * u;
  }

  /**
   * @brief Compute the Jacobians of the floating base propeller actuation
   * function
   *
   * @param[in] data  Floating base propellers actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
#ifndef NDEBUG
  virtual void calcDiff(const boost::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) {
#else
  virtual void calcDiff(const boost::shared_ptr<Data>&,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) {
#endif
    // The derivatives has constant values which were set in createData.
    assert_pretty(MatrixXs(data->dtau_du).isApprox(W_thrust_),
                  "dtau_du has wrong value");
  }

  virtual void commands(const boost::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>& tau) {
    data->u.noalias() = data->Mtau * tau;
  }

#ifndef NDEBUG
  virtual void torqueTransform(const boost::shared_ptr<Data>& data,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#else
  virtual void torqueTransform(const boost::shared_ptr<Data>&,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#endif
    // The torque transform has constant values which were set in createData.
    assert_pretty(MatrixXs(data->Mtau).isApprox(Mtau_), "Mtau has wrong value");
  }

  /**
   * @brief Create the floating base propeller actuation data
   *
   * @return the actuation data
   */
  virtual boost::shared_ptr<Data> createData() {
    boost::shared_ptr<Data> data =
        boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    updateData(data);
    return data;
  }

  /**
   * @brief Return the vector of propellers
   */
  const std::vector<Propeller>& get_propellers() const { return propellers_; }

  /**
   * @brief Return the number of propellers
   */
  std::size_t get_npropellers() const { return n_propellers_; }

  /**
   * @brief Modify the vector of propellers
   *
   * Since we don't want to allocate data, we request to pass the same
   * number of propellers.
   *
   * @param[in] propellers  Vector of propellers
   */
  void set_propellers(const std::vector<Propeller>& propellers) {
    if (static_cast<std::size_t>(propellers.size()) != n_propellers_) {
      throw_pretty("Invalid argument: "
                   << "the number of propellers is wrong (it should be " +
                          std::to_string(n_propellers_) + ")");
    }
    propellers_ = propellers;
    // Update the mapping matrix from propellers thrust to body force/moments
    for (std::size_t i = 0; i < n_propellers_; ++i) {
      const Propeller& p = propellers_[i];
      const Vector3s& f_z = p.pose.rotation() * Vector3s::UnitZ();
      W_thrust_.template topRows<3>().col(i) += f_z;
      W_thrust_.template middleRows<3>(3).col(i).noalias() +=
          p.pose.translation().cross(Vector3s::UnitZ());
      switch (p.type) {
        case CW:
          W_thrust_.template middleRows<3>(3).col(i) +=
              (p.ctorque / p.cthrust) * f_z;
          break;
        case CCW:
          W_thrust_.template middleRows<3>(3).col(i) -=
              (p.ctorque / p.cthrust) * f_z;
          break;
      }
    }
    // Compute the torque transform matrix from generalized torques to joint
    // torque inputs
    Mtau_ = pseudoInverse(MatrixXs(W_thrust_));
    S_.noalias() = W_thrust_ * Mtau_;
    update_data_ = true;
  }

  const MatrixXs& get_Wthrust() const { return W_thrust_; }

  const MatrixXs& get_S() const { return S_; }

  void print(std::ostream& os) const {
    os << "ActuationModelFloatingBasePropellers {nu=" << nu_
       << ", npropellers=" << n_propellers_ << ", propellers=" << std::endl;
    for (std::size_t i = 0; i < n_propellers_; ++i) {
      os << std::to_string(i) << ": " << propellers_[i];
    }
    os << "}";
  }

 protected:
  std::vector<Propeller> propellers_;  //!< Vector of propellers
  std::size_t n_propellers_;           //!< Number of propellers
  MatrixXs W_thrust_;  //!< Matrix from propellers thrusts to body wrench
  MatrixXs Mtau_;  //!< Constaint torque transform from generalized torques to
                   //!< joint torque inputs
  MatrixXs S_;     //!< Selection matrix for under-actuation part

  bool update_data_;
  using Base::nu_;
  using Base::state_;

 private:
  void updateData(const boost::shared_ptr<Data>& data) {
    data->dtau_du = W_thrust_;
    data->Mtau = Mtau_;
    const std::size_t nv = state_->get_nv();
    for (std::size_t k = 0; k < nv; ++k) {
      if (fabs(S_(k, k)) < std::numeric_limits<Scalar>::epsilon()) {
        data->tau_set[k] = false;
      } else {
        data->tau_set[k] = true;
      }
    }
    update_data_ = false;
  }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_PROPELLERS_HPP_
