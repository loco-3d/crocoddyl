///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACT_HPP_
#define CROCODDYL_MULTIBODY_CONTACT_HPP_

#include <array>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/implicit-constraint-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModelTpl : public ImplicitConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ImplicitConstraintModelBase, ContactModelTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImplicitConstraintModelAbstractTpl<Scalar> Base;
  typedef ContactDataTpl<Scalar> Data;
  typedef ImplicitConstraintDataAbstractTpl<Scalar>
      ImplicitConstraintDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef std::array<bool, 6> MaskArray;

  /**
   * @brief Initialize the masked contact model
   *
   * This model generalizes 1D/2D/3D/6D contacts through a 6D boolean mask.
   *
   * @param[in] state      State of the multibody system
   * @param[in] id         Reference frame id of the contact
   * @param[in] reference  Contact placement used for Baumgarte stabilization
   * @param[in] type       Type of implicit constraint
   * @param[in] nu         Dimension of control vector
   * @param[in] gains      Baumgarte stabilization gains
   * @param[in] mask       Selection mask of constrained spatial axes
   */
  ContactModelTpl(std::shared_ptr<StateMultibody> state,
                  const pinocchio::FrameIndex id, const SE3& reference,
                  const pinocchio::ReferenceFrame type, const std::size_t nu,
                  const Vector2s& gains = Vector2s::Zero(),
                  const MaskArray& mask = DefaultMask());

  /**
   * @brief Initialize the masked contact model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state      State of the multibody system
   * @param[in] id         Reference frame id of the contact
   * @param[in] reference  Contact placement used for Baumgarte stabilization
   * @param[in] type       Type of implicit constraint
   * @param[in] gains      Baumgarte stabilization gains
   * @param[in] mask       Selection mask of constrained spatial axes
   */
  ContactModelTpl(std::shared_ptr<StateMultibody> state,
                  const pinocchio::FrameIndex id, const SE3& reference,
                  const pinocchio::ReferenceFrame type,
                  const Vector2s& gains = Vector2s::Zero(),
                  const MaskArray& mask = DefaultMask());
  virtual ~ContactModelTpl() = default;

  /**
   * @brief Compute the masked contact Jacobian and acceleration drift
   *
   * @param[in] data  Contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute derivatives of the masked contact holonomic constraint
   *
   * For WORLD / LOCAL_WORLD_ALIGNED frames, this includes frame-rotation terms
   * in the same spirit as the impulse/contact 6D implementations.
   *
   * @param[in] data  Contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Convert the masked force into a stack of spatial forces
   *
   * For WORLD / LOCAL_WORLD_ALIGNED frames, this also updates the skew-term
   * contribution to `dtau_dq`.
   *
   * @param[in] data   Contact data
   * @param[in] force  Masked contact force
   */
  virtual void updateForce(
      const std::shared_ptr<ImplicitConstraintDataAbstract>& data,
      const VectorXs& force) override;

  /**
   * @brief Create the masked contact data
   */
  virtual std::shared_ptr<ImplicitConstraintDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) override;

  /**
   * @brief Cast the masked contact model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ContactModelTpl<NewScalar> A contact model with the new scalar
   * type.
   */
  template <typename NewScalar>
  ContactModelTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference frame placement
   */
  const SE3& get_reference() const;

  /**
   * @brief Modify the reference frame placement
   */
  void set_reference(const SE3& reference);

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
   * @brief Return the 6D selection matrix
   */
  const Matrix6xs& get_selection_matrix() const;

  /**
   * @brief Print relevant information of the masked contact model
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
  SE3 reference_;   //!< Contact placement used for Baumgarte stabilization
  Vector2s gains_;  //!< Baumgarte stabilization gains
  MaskArray mask_;  //!< Selection mask of constrained spatial axes
  bool spatial_;    //!< Use spatial rather than classical acceleration
  Matrix6xs S_;     //!< Selection matrix from 6D to masked coordinates
};

/**
 * @brief Data structure for masked contact models
 */
template <typename _Scalar>
struct ContactDataTpl : public ImplicitConstraintDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImplicitConstraintDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::MotionTpl<Scalar> Motion;
  typedef typename pinocchio::ForceTpl<Scalar> Force;
  typedef typename ContactModelTpl<Scalar>::MaskArray MaskArray;

  /**
   * @brief Initialize the masked contact data
   *
   * @param[in] model  Masked contact model
   * @param[in] data   Pinocchio data
   */
  template <template <typename Scalar> class Model>
  ContactDataTpl(Model<Scalar>* const model,
                 pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        mask(model->get_mask()),
        rMf(SE3::Identity()),
        lwaMl(SE3::Identity()),
        v(Motion::Zero()),
        v0(Motion::Zero()),
        a0_local(Motion::Zero()),
        f_local(Force::Zero()),
        dp(Vector3s::Zero()),
        dp_local(Vector3s::Zero()),
        a0_6d(Vector6s::Zero()),
        force_6d(Vector6s::Zero()),
        Jc_6d(6, model->get_state()->get_nv()),
        da0_dx_6d(6, model->get_state()->get_ndx()),
        dv0_dq_6d(6, model->get_state()->get_nv()),
        da0_local_dx(6, model->get_state()->get_ndx()),
        dv0_local_dq(6, model->get_state()->get_nv()),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fJf_df(6, model->get_state()->get_nv()) {
    this->frame = model->get_id();
    this->jMf =
        model->get_state()->get_pinocchio()->frames[this->frame].placement;
    this->fXj = this->jMf.inverse().toActionMatrix();
    Jc_6d.setZero();
    da0_dx_6d.setZero();
    dv0_dq_6d.setZero();
    da0_local_dx.setZero();
    dv0_local_dq.setZero();
    fJf.setZero();
    v_partial_dq.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    av_world_skew.setZero();
    aw_world_skew.setZero();
    av_skew.setZero();
    aw_skew.setZero();
    vv_world_skew.setZero();
    vw_world_skew.setZero();
    vv_skew.setZero();
    vw_skew.setZero();
    fv_skew.setZero();
    fw_skew.setZero();
    dp_skew.setZero();
    rMf_Jlog6.setZero();
    fJf_df.setZero();
  }
  virtual ~ContactDataTpl() = default;

  using Base::a0;
  using Base::da0_dx;
  using Base::df_du;
  using Base::df_dx;
  using Base::dtau_dq;
  using Base::dv0_dq;
  using Base::f;
  using Base::fext;
  using Base::frame;
  using Base::fXj;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;

  const MaskArray mask;  //!< Immutable selected spatial-coordinate layout
  SE3 rMf;               //!< Relative reference error placement
  SE3 lwaMl;             //!< Local-world-aligned transform
  Motion v;              //!< Frame spatial velocity
  Motion v0;             //!< Frame spatial velocity in selected frame type
  Motion a0_local;       //!< Local desired acceleration drift
  Force f_local;         //!< Local force representation
  Vector3s dp;           //!< Translational reference error in world coordinates
  Vector3s dp_local;     //!< Translational reference error in local coordinates
  Vector6s a0_6d;        //!< Full 6D acceleration drift
  Vector6s force_6d;     //!< Full 6D force
  Matrix6xs Jc_6d;       //!< Full 6D contact Jacobian
  Matrix6xs da0_dx_6d;   //!< Full 6D Jacobian of acceleration drift wrt state
  Matrix6xs dv0_dq_6d;   //!< Full 6D Jacobian of velocity drift wrt q
  Matrix6xs da0_local_dx;
  Matrix6xs dv0_local_dq;
  Matrix6xs fJf;
  Matrix6xs v_partial_dq;
  Matrix6xs a_partial_dq;
  Matrix6xs a_partial_dv;
  Matrix6xs a_partial_da;
  Matrix3s av_world_skew;
  Matrix3s aw_world_skew;
  Matrix3s av_skew;
  Matrix3s aw_skew;
  Matrix3s vv_world_skew;
  Matrix3s vw_world_skew;
  Matrix3s vv_skew;
  Matrix3s vw_skew;
  Matrix3s fv_skew;
  Matrix3s fw_skew;
  Matrix3s dp_skew;
  Matrix6s rMf_Jlog6;
  Matrix6xs fJf_df;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/implicit-constraints/contact.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ContactModelTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ContactDataTpl)

#endif  // CROCODDYL_MULTIBODY_CONTACT_HPP_
