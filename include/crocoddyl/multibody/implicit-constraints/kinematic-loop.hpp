///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_KINEMATIC_LOOP_HPP_
#define CROCODDYL_MULTIBODY_KINEMATIC_LOOP_HPP_

#include <array>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/implicit-constraint-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class KinematicLoopModelTpl
    : public ImplicitConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ImplicitConstraintModelBase, KinematicLoopModelTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImplicitConstraintModelAbstractTpl<Scalar> Base;
  typedef KinematicLoopDataTpl<Scalar> Data;
  typedef ImplicitConstraintDataAbstractTpl<Scalar>
      ImplicitConstraintDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef std::array<bool, 6> MaskArray;

  /**
   * @brief Initialize the kinematic-loop model
   *
   * @param[in] state             State of the multibody system
   * @param[in] joint1_id         First loop joint id
   * @param[in] joint1_placement  First loop frame placement in joint-1 frame
   * @param[in] joint2_id         Second loop joint id
   * @param[in] joint2_placement  Second loop frame placement in joint-2 frame
   * @param[in] type              Type of implicit constraint
   * @param[in] nu                Dimension of control vector
   * @param[in] gains             Baumgarte stabilization gains
   * @param[in] mask              Selection mask of constrained dimensions
   */
  KinematicLoopModelTpl(std::shared_ptr<StateMultibody> state,
                        const pinocchio::JointIndex joint1_id,
                        const SE3& joint1_placement,
                        const pinocchio::JointIndex joint2_id,
                        const SE3& joint2_placement,
                        const pinocchio::ReferenceFrame type,
                        const std::size_t nu,
                        const Vector2s& gains = Vector2s::Zero(),
                        const MaskArray& mask = DefaultMask());

  /**
   * @brief Initialize the kinematic-loop model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state             State of the multibody system
   * @param[in] joint1_id         First loop joint id
   * @param[in] joint1_placement  First loop frame placement in joint-1 frame
   * @param[in] joint2_id         Second loop joint id
   * @param[in] joint2_placement  Second loop frame placement in joint-2 frame
   * @param[in] type              Type of implicit constraint
   * @param[in] gains             Baumgarte stabilization gains
   * @param[in] mask              Selection mask of constrained dimensions
   */
  KinematicLoopModelTpl(std::shared_ptr<StateMultibody> state,
                        const pinocchio::JointIndex joint1_id,
                        const SE3& joint1_placement,
                        const pinocchio::JointIndex joint2_id,
                        const SE3& joint2_placement,
                        const pinocchio::ReferenceFrame type,
                        const Vector2s& gains = Vector2s::Zero(),
                        const MaskArray& mask = DefaultMask());
  virtual ~KinematicLoopModelTpl() = default;

  /**
   * @brief Compute the kinematic-loop Jacobian and acceleration drift
   *
   * @param[in] data  Kinematic-loop data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the kinematic-loop holonomic constraint
   *
   * @param[in] data  Kinematic-loop data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   Kinematic-loop data
   * @param[in] force  Kinematic-loop force
   */
  virtual void updateForce(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const VectorXs& force) override;

  /**
   * @brief Update the stack of loop-force Jacobians
   *
   * @param[in] data   Kinematic-loop data
   * @param[in] df_dx  Jacobian of loop force with respect to state
   * @param[in] df_du  Jacobian of loop force with respect to control
   */
  virtual void updateForceDiff(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& df_dx,
      const Eigen::Ref<const MatrixXs>& df_du) const override;

  /**
   * @brief Create the kinematic-loop data
   */
  virtual std::shared_ptr<ImplicitConstraintDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) override;

  /**
   * @brief Cast the kinematic-loop model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return KinematicLoopModelTpl<NewScalar> A kinematic-loop model with the
   * new scalar type.
   */
  template <typename NewScalar>
  KinematicLoopModelTpl<NewScalar> cast() const;

  /**
   * @brief Return the first loop joint id
   */
  const pinocchio::JointIndex& get_joint1_id() const;

  /**
   * @brief Return the second loop joint id
   */
  const pinocchio::JointIndex& get_joint2_id() const;

  /**
   * @brief Return the first loop frame placement
   */
  const SE3& get_joint1_placement() const;

  /**
   * @brief Return the second loop frame placement
   */
  const SE3& get_joint2_placement() const;

  /**
   * @brief Return the Baumgarte stabilization gains
   */
  const Vector2s& get_gains() const;

  /**
   * @brief Modify the Baumgarte stabilization gains
   */
  void set_gains(const Vector2s& gains);

  /**
   * @brief Return the selection mask
   */
  const MaskArray& get_mask() const;

  /**
   * @brief Return the loop selection matrix
   */
  const Matrix6xs& get_selection_matrix() const;

  /**
   * @brief Print relevant information of the kinematic-loop model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

  /**
   * @brief Return the default selection mask
   */
  static MaskArray DefaultMask();

 protected:
  using Base::id_;
  using Base::nc_;
  using Base::nu_;
  using Base::state_;
  using Base::type_;

 private:
  pinocchio::JointIndex joint1_id_;  //!< First loop joint id
  SE3 joint1_placement_;             //!< First loop frame placement
  pinocchio::JointIndex joint2_id_;  //!< Second loop joint id
  SE3 joint2_placement_;             //!< Second loop frame placement
  Vector2s gains_;                   //!< Baumgarte stabilization gains
  MaskArray mask_;                   //!< Selection mask of constrained axes
  Matrix6xs S_;                      //!< Loop selection matrix
};

template <typename _Scalar>
struct KinematicLoopDataTpl
    : public ImplicitConstraintDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImplicitConstraintDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::MotionTpl<Scalar> Motion;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  /**
   * @brief Initialize the kinematic-loop data
   *
   * @param[in] model  Kinematic-loop model
   * @param[in] data   Pinocchio data
   */
  template <template <typename Scalar> class Model>
  KinematicLoopDataTpl(Model<Scalar>* const model,
                       pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        a0_6d(Vector6s::Zero()),
        force_6d(Vector6s::Zero()),
        da0_dx_6d(6, model->get_state()->get_ndx()),
        dv0_dq_6d(6, model->get_state()->get_nv()),
        j1Jj1(6, model->get_state()->get_nv()),
        j2Jj2(6, model->get_state()->get_nv()),
        j2Jj1(6, model->get_state()->get_nv()),
        f1Jf1(6, model->get_state()->get_nv()),
        f2Jf2(6, model->get_state()->get_nv()),
        f1Jf2(6, model->get_state()->get_nv()),
        v1_partial_dq(6, model->get_state()->get_nv()),
        v2_partial_dq(6, model->get_state()->get_nv()),
        f1_v1_partial_dq(6, model->get_state()->get_nv()),
        f2_v2_partial_dq(6, model->get_state()->get_nv()),
        f1_v2_partial_dq(6, model->get_state()->get_nv()),
        a1_partial_dq(6, model->get_state()->get_nv()),
        a2_partial_dq(6, model->get_state()->get_nv()),
        a1_partial_dv(6, model->get_state()->get_nv()),
        a2_partial_dv(6, model->get_state()->get_nv()),
        a1_partial_da(6, model->get_state()->get_nv()),
        a2_partial_da(6, model->get_state()->get_nv()),
        f2_a2_partial_dq(6, model->get_state()->get_nv()),
        f2_a2_partial_dv(6, model->get_state()->get_nv()),
        da0_dq_t1(6, model->get_state()->get_nv()),
        da0_dq_t2(6, model->get_state()->get_nv()),
        da0_dq_t3(6, model->get_state()->get_nv()),
        da0_dq_t3_tmp(6, model->get_state()->get_nv()),
        dpos_dq(6, model->get_state()->get_nv()),
        dvel_dq(6, model->get_state()->get_nv()),
        dtau_dq_tmp(6, model->get_state()->get_nv()),
        f1Xj1(model->get_joint1_placement().inverse().toActionMatrix()),
        f2Xj2(model->get_joint2_placement().inverse().toActionMatrix()),
        f1Xf2(Matrix6s::Identity()),
        f_cross(Matrix6s::Zero()),
        Jlog6(Matrix6s::Zero()),
        oMf1(model->get_joint1_placement()),
        oMf2(model->get_joint2_placement()),
        f1Mf2(model->get_joint1_placement().actInv(
            model->get_joint2_placement())),
        j2Mj1(SE3::Identity()),
        f1vf1(Motion::Zero()),
        f2vf2(Motion::Zero()),
        f1vf2(Motion::Zero()),
        f1af1(Motion::Zero()),
        f2af2(Motion::Zero()),
        f1af2(Motion::Zero()),
        joint1_f(Force::Zero()),
        joint2_f(Force::Zero()) {
    da0_dx_6d.setZero();
    dv0_dq_6d.setZero();
    j1Jj1.setZero();
    j2Jj2.setZero();
    j2Jj1.setZero();
    f1Jf1.setZero();
    f2Jf2.setZero();
    f1Jf2.setZero();
    v1_partial_dq.setZero();
    v2_partial_dq.setZero();
    f1_v1_partial_dq.setZero();
    f2_v2_partial_dq.setZero();
    f1_v2_partial_dq.setZero();
    a1_partial_dq.setZero();
    a2_partial_dq.setZero();
    a1_partial_dv.setZero();
    a2_partial_dv.setZero();
    a1_partial_da.setZero();
    a2_partial_da.setZero();
    f2_a2_partial_dq.setZero();
    f2_a2_partial_dv.setZero();
    da0_dq_t1.setZero();
    da0_dq_t2.setZero();
    da0_dq_t3.setZero();
    da0_dq_t3_tmp.setZero();
    dpos_dq.setZero();
    dvel_dq.setZero();
    dtau_dq_tmp.setZero();
  }
  virtual ~KinematicLoopDataTpl() = default;

  using Base::a0;
  using Base::da0_dx;
  using Base::df_du;
  using Base::df_dx;
  using Base::dtau_dq;
  using Base::dv0_dq;
  using Base::f;
  using Base::fext;
  using Base::Jc;
  using Base::pinocchio;

  Vector6s a0_6d;      //!< Full 6D loop acceleration drift
  Vector6s force_6d;   //!< Full 6D loop force
  MatrixXs da0_dx_6d;  //!< Full 6D drift Jacobian wrt state
  MatrixXs dv0_dq_6d;  //!< Full 6D velocity-drift Jacobian wrt q
  Matrix6xs j1Jj1;
  Matrix6xs j2Jj2;
  Matrix6xs j2Jj1;
  Matrix6xs f1Jf1;
  Matrix6xs f2Jf2;
  Matrix6xs f1Jf2;
  Matrix6xs v1_partial_dq;
  Matrix6xs v2_partial_dq;
  Matrix6xs f1_v1_partial_dq;
  Matrix6xs f2_v2_partial_dq;
  Matrix6xs f1_v2_partial_dq;
  Matrix6xs a1_partial_dq;
  Matrix6xs a2_partial_dq;
  Matrix6xs a1_partial_dv;
  Matrix6xs a2_partial_dv;
  Matrix6xs a1_partial_da;
  Matrix6xs a2_partial_da;
  Matrix6xs f2_a2_partial_dq;
  Matrix6xs f2_a2_partial_dv;
  Matrix6xs da0_dq_t1;
  Matrix6xs da0_dq_t2;
  Matrix6xs da0_dq_t3;
  Matrix6xs da0_dq_t3_tmp;
  Matrix6xs dpos_dq;
  Matrix6xs dvel_dq;
  Matrix6xs dtau_dq_tmp;
  Matrix6s f1Xj1;
  Matrix6s f2Xj2;
  Matrix6s f1Xf2;
  Matrix6s f_cross;
  Matrix6s Jlog6;
  SE3 oMf1;
  SE3 oMf2;
  SE3 f1Mf2;
  SE3 j2Mj1;
  Motion f1vf1;
  Motion f2vf2;
  Motion f1vf2;
  Motion f1af1;
  Motion f2af2;
  Motion f1af2;
  Force joint1_f;
  Force joint2_f;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/implicit-constraints/kinematic-loop.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::KinematicLoopModelTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::KinematicLoopDataTpl)

#endif  // CROCODDYL_MULTIBODY_KINEMATIC_LOOP_HPP_
