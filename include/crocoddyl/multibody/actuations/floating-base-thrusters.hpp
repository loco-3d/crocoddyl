///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2014-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_PROPELLERS_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_PROPELLERS_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

enum ThrusterType { CW = 0, CCW };

template <typename _Scalar>
struct ThrusterTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef ThrusterTpl<Scalar> Thruster;

  /**
   * @brief Initialize the thruster in a give pose from the root joint.
   *
   * @param pose[in]     Pose from root joint
   * @param ctorque[in]  Coefficient of generated torque per thrust
   * @param type[in]     Type of thruster (clockwise or counterclockwise,
   * default clockwise)
   * @param[in] min_thrust[in]  Minimum thrust (default 0.)
   * @param[in] max_thrust[in]  Maximum thrust (default inf number))
   */
  ThrusterTpl(const SE3& pose, const Scalar ctorque,
              const ThrusterType type = CW,
              const Scalar min_thrust = Scalar(0.),
              const Scalar max_thrust = std::numeric_limits<Scalar>::infinity())
      : pose(pose),
        ctorque(ctorque),
        type(type),
        min_thrust(min_thrust),
        max_thrust(max_thrust) {}

  /**
   * @brief Initialize the thruster in a pose in the origin of the root joint.
   *
   * @param pose[in]     Pose from root joint
   * @param ctorque[in]  Coefficient of generated torque per thrust
   * @param type[in]     Type of thruster (clockwise or counterclockwise,
   * default clockwise)
   * @param[in] min_thrust[in]  Minimum thrust (default 0.)
   * @param[in] max_thrust[in]  Maximum thrust (default inf number))
   */
  ThrusterTpl(const Scalar ctorque, const ThrusterType type = CW,
              const Scalar min_thrust = Scalar(0.),
              const Scalar max_thrust = std::numeric_limits<Scalar>::infinity())
      : pose(SE3::Identity()),
        ctorque(ctorque),
        type(type),
        min_thrust(min_thrust),
        max_thrust(max_thrust) {}
  ThrusterTpl(const ThrusterTpl<Scalar>& clone)
      : pose(clone.pose),
        ctorque(clone.ctorque),
        type(clone.type),
        min_thrust(clone.min_thrust),
        max_thrust(clone.max_thrust) {}

  template <typename NewScalar>
  ThrusterTpl<NewScalar> cast() const {
    typedef ThrusterTpl<NewScalar> ReturnType;
    ReturnType ret(
        pose.template cast<NewScalar>(), scalar_cast<NewScalar>(ctorque), type,
        scalar_cast<NewScalar>(min_thrust), scalar_cast<NewScalar>(max_thrust));
    return ret;
  }

  ThrusterTpl& operator=(const ThrusterTpl<Scalar>& other) {
    if (this != &other) {
      pose = other.pose;
      ctorque = other.ctorque;
      type = other.type;
      min_thrust = other.min_thrust;
      max_thrust = other.max_thrust;
    }
    return *this;
  }

  template <typename OtherScalar>
  bool operator==(const ThrusterTpl<OtherScalar>& other) const {
    return (pose == other.pose && ctorque == other.ctorque &&
            type == other.type && min_thrust == other.min_thrust &&
            max_thrust == other.max_thrust);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ThrusterTpl<Scalar>& X) {
    os << "      pose:" << std::endl
       << X.pose << "   ctorque: " << X.ctorque << std::endl
       << "      type: " << X.type << std::endl
       << "min_thrust: " << X.min_thrust << std::endl
       << "max_thrust: " << X.max_thrust << std::endl;
    return os;
  }

  SE3 pose;           //!< Thruster pose
  Scalar ctorque;     //!< Coefficient of generated torque per thrust
  ThrusterType type;  //!< Type of thruster (CW and CCW for clockwise and
                      //!< counterclockwise, respectively)
  Scalar min_thrust;  //!< Minimum thrust
  Scalar max_thrust;  //!< Minimum thrust
};

/**
 * @brief Actuation models for floating base systems actuated with thrusters
 *
 * This actuation model models floating base robots equipped with thrusters,
 * e.g., multicopters or marine robots equipped with manipulators. It control
 * inputs are the thrusters' thrust (i.e., forces) and joint torques.
 *
 * Both actuation and Jacobians are computed analytically by `calc` and
 * `calcDiff`, respectively.
 *
 * We assume the robot velocity to zero for easily related square thruster
 * velocities with thrust and torque generated. This approach is similarly
 * implemented in M. Geisert and N. Mansard, "Trajectory generation for
 * quadrotor based systems using numerical optimal control", (ICRA). See Section
 * III.C.
 *
 * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelFloatingBaseThrustersTpl
    : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase,
                         ActuationModelFloatingBaseThrustersTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ThrusterTpl<Scalar> Thruster;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the floating base actuation model equipped with
   * thrusters
   *
   * @param[in] state      State of the dynamical system
   * @param[in] thrusters  Vector of thrusters
   */
  ActuationModelFloatingBaseThrustersTpl(std::shared_ptr<StateMultibody> state,
                                         const std::vector<Thruster>& thrusters)
      : Base(state,
             state->get_nv() -
                 state->get_pinocchio()
                     ->joints[(
                         state->get_pinocchio()->existJointName("root_joint")
                             ? state->get_pinocchio()->getJointId("root_joint")
                             : 0)]
                     .nv() +
                 thrusters.size()),
        thrusters_(thrusters),
        n_thrusters_(thrusters.size()),
        W_thrust_(state_->get_nv(), nu_),
        update_data_(true) {
    if (!state->get_pinocchio()->existJointName("root_joint")) {
      throw_pretty(
          "Invalid argument: "
          << "the first joint has to be a root one (e.g., free-flyer joint)");
    }
    // Update the joint actuation part
    W_thrust_.setZero();
    if (nu_ > n_thrusters_) {
      W_thrust_.bottomRightCorner(nu_ - n_thrusters_, nu_ - n_thrusters_)
          .diagonal()
          .setOnes();
    }
    // Update the floating base actuation part
    set_thrusters(thrusters_);
  }
  virtual ~ActuationModelFloatingBaseThrustersTpl() = default;

  /**
   * @brief Compute the actuation signal and actuation set from its thrust
   * and joint torque inputs \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   *
   * @param[in] data  Floating base thrusters actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<Data>& data,
                    const Eigen::Ref<const VectorXs>&,
                    const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    if (update_data_) {
      updateData(data);
    }
    data->tau.noalias() = data->dtau_du * u;
  }

  /**
   * @brief Compute the Jacobians of the floating base thruster actuation
   * function
   *
   * @param[in] data  Floating base thrusters actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
#ifndef NDEBUG
  virtual void calcDiff(const std::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) override {
#else
  virtual void calcDiff(const std::shared_ptr<Data>&,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>&) override {
#endif
    // The derivatives has constant values which were set in createData.
    assert_pretty(MatrixXs(data->dtau_du).isApprox(W_thrust_),
                  "dtau_du has wrong value");
  }

  virtual void commands(const std::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>& tau) override {
    data->u.noalias() = data->Mtau * tau;
  }

#ifndef NDEBUG
  virtual void torqueTransform(const std::shared_ptr<Data>& data,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) override {
#else
  virtual void torqueTransform(const std::shared_ptr<Data>&,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) override {
#endif
    // The torque transform has constant values which were set in createData.
    assert_pretty(MatrixXs(data->Mtau).isApprox(Mtau_), "Mtau has wrong value");
  }

  /**
   * @brief Create the floating base thruster actuation data
   *
   * @return the actuation data
   */
  virtual std::shared_ptr<Data> createData() override {
    std::shared_ptr<Data> data =
        std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    updateData(data);
    return data;
  }

  template <typename NewScalar>
  ActuationModelFloatingBaseThrustersTpl<NewScalar> cast() const {
    typedef ActuationModelFloatingBaseThrustersTpl<NewScalar> ReturnType;
    typedef StateMultibodyTpl<NewScalar> StateType;
    typedef ThrusterTpl<NewScalar> ThrusterType;
    std::vector<ThrusterType> thrusters = vector_cast<NewScalar>(thrusters_);
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        thrusters);
    return ret;
  }

  /**
   * @brief Return the vector of thrusters
   */
  const std::vector<Thruster>& get_thrusters() const { return thrusters_; }

  /**
   * @brief Return the number of thrusters
   */
  std::size_t get_nthrusters() const { return n_thrusters_; }

  /**
   * @brief Modify the vector of thrusters
   *
   * Since we don't want to allocate data, we request to pass the same
   * number of thrusters.
   *
   * @param[in] thrusters  Vector of thrusters
   */
  void set_thrusters(const std::vector<Thruster>& thrusters) {
    if (static_cast<std::size_t>(thrusters.size()) != n_thrusters_) {
      throw_pretty("Invalid argument: "
                   << "the number of thrusters is wrong (it should be " +
                          std::to_string(n_thrusters_) + ")");
    }
    thrusters_ = thrusters;
    // Update the mapping matrix from thrusters thrust to body force/moments
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
      const Thruster& p = thrusters_[i];
      const Vector3s& f_z = p.pose.rotation() * Vector3s::UnitZ();
      W_thrust_.template topRows<3>().col(i) += f_z;
      W_thrust_.template middleRows<3>(3).col(i).noalias() +=
          p.pose.translation().cross(f_z);
      switch (p.type) {
        case CW:
          W_thrust_.template middleRows<3>(3).col(i) += p.ctorque * f_z;
          break;
        case CCW:
          W_thrust_.template middleRows<3>(3).col(i) -= p.ctorque * f_z;
          break;
      }
    }
    // Compute the torque transform matrix from generalized torques to joint
    // torque inputs
    Mtau_ = pseudoInverse(W_thrust_);
    S_.noalias() = W_thrust_ * Mtau_;
    update_data_ = true;
  }

  const MatrixXs& get_Wthrust() const { return W_thrust_; }

  const MatrixXs& get_S() const { return S_; }

  void print(std::ostream& os) const override {
    os << "ActuationModelFloatingBaseThrusters {nu=" << nu_
       << ", nthrusters=" << n_thrusters_ << ", thrusters=" << std::endl;
    for (std::size_t i = 0; i < n_thrusters_; ++i) {
      os << std::to_string(i) << ": " << thrusters_[i];
    }
    os << "}";
  }

 protected:
  std::vector<Thruster> thrusters_;  //!< Vector of thrusters
  std::size_t n_thrusters_;          //!< Number of thrusters
  MatrixXs W_thrust_;  //!< Matrix from thrusters thrusts to body wrench
  MatrixXs Mtau_;  //!< Constaint torque transform from generalized torques to
                   //!< joint torque inputs
  MatrixXs S_;     //!< Selection matrix for under-actuation part

  bool update_data_;
  using Base::nu_;
  using Base::state_;

 private:
  void updateData(const std::shared_ptr<Data>& data) {
    data->dtau_du = W_thrust_;
    data->Mtau = Mtau_;
    const std::size_t nv = state_->get_nv();
    for (std::size_t k = 0; k < nv; ++k) {
      data->tau_set[k] = if_static(S_(k, k));
    }
    update_data_ = false;
  }

  // Use for floating-point types
  template <typename Scalar>
  typename std::enable_if<std::is_floating_point<Scalar>::value, bool>::type
  if_static(const Scalar& condition) {
    return (fabs(condition) < std::numeric_limits<Scalar>::epsilon()) ? false
                                                                      : true;
  }

#ifdef CROCODDYL_WITH_CODEGEN
  // Use for CppAD types
  template <typename Scalar>
  typename std::enable_if<!std::is_floating_point<Scalar>::value, bool>::type
  if_static(const Scalar& condition) {
    return CppAD::Value(CppAD::fabs(condition)) >=
           CppAD::numeric_limits<Scalar>::epsilon();
  }
#endif
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ThrusterTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActuationModelFloatingBaseThrustersTpl)

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_PROPELLERS_HPP_
