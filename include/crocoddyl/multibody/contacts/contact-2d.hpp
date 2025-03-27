///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_2D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_2D_HPP_

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModel2DTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ContactModelBase, ContactModel2DTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactData2DTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the 2d contact model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id of the contact
   * @param[in] xref   Contact position used for the Baumgarte stabilization
   * @param[in] nu     Dimension of the control vector
   * @param[in] gains  Baumgarte stabilization gains
   */
  ContactModel2DTpl(std::shared_ptr<StateMultibody> state,
                    const pinocchio::FrameIndex id, const Vector2s& xref,
                    const std::size_t nu,
                    const Vector2s& gains = Vector2s::Zero());

  /**
   * @brief Initialize the 2d contact model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id of the contact
   * @param[in] xref   Contact position used for the Baumgarte stabilization
   * @param[in] gains  Baumgarte stabilization gains
   */
  ContactModel2DTpl(std::shared_ptr<StateMultibody> state,
                    const pinocchio::FrameIndex id, const Vector2s& xref,
                    const Vector2s& gains = Vector2s::Zero());
  virtual ~ContactModel2DTpl() = default;

  /**
   * @brief Compute the 2d contact Jacobian and drift
   *
   * @param[in] data  2d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the 2d contact holonomic constraint
   *
   * @param[in] data  2d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ContactDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   2d contact data
   * @param[in] force  2d force
   */
  virtual void updateForce(const std::shared_ptr<ContactDataAbstract>& data,
                           const VectorXs& force) override;

  /**
   * @brief Create the 2d contact data
   */
  virtual std::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) override;

  /**
   * @brief Cast the contact-2d model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ContactModel2DTpl<NewScalar> A contact model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ContactModel2DTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference frame translation
   */
  const Vector2s& get_reference() const;

  /**
   * @brief Create the 2d contact data
   */
  const Vector2s& get_gains() const;

  /**
   * @brief Modify the reference frame translation
   */
  void set_reference(const Vector2s& reference);

  /**
   * @brief Print relevant information of the 2d contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::id_;
  using Base::nc_;
  using Base::nu_;
  using Base::state_;

 private:
  Vector2s xref_;   //!< Contact position used for the Baumgarte stabilization
  Vector2s gains_;  //!< Baumgarte stabilization gains
};

template <typename _Scalar>
struct ContactData2DTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Matrix2s Matrix2s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;

  template <template <typename Scalar> class Model>
  ContactData2DTpl(Model<Scalar>* const model,
                   pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fXjdv_dq(6, model->get_state()->get_nv()),
        fXjda_dq(6, model->get_state()->get_nv()),
        fXjda_dv(6, model->get_state()->get_nv()) {
    frame = model->get_id();
    jMf = model->get_state()->get_pinocchio()->frames[frame].placement;
    fXj = jMf.inverse().toActionMatrix();
    fJf.setZero();
    v_partial_dq.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    fXjdv_dq.setZero();
    fXjda_dq.setZero();
    fXjda_dv.setZero();
    vv.setZero();
    vw.setZero();
    vv_skew.setZero();
    vw_skew.setZero();
    oRf.setZero();
  }
  virtual ~ContactData2DTpl() = default;

  using Base::a0;
  using Base::da0_dx;
  using Base::df_du;
  using Base::df_dx;
  using Base::f;
  using Base::frame;
  using Base::fXj;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;

  pinocchio::MotionTpl<Scalar> v;
  pinocchio::MotionTpl<Scalar> a;
  Matrix6xs fJf;
  Matrix6xs v_partial_dq;
  Matrix6xs a_partial_dq;
  Matrix6xs a_partial_dv;
  Matrix6xs a_partial_da;
  Matrix6xs fXjdv_dq;
  Matrix6xs fXjda_dq;
  Matrix6xs fXjda_dv;
  Vector3s vv;
  Vector3s vw;
  Matrix3s vv_skew;
  Matrix3s vw_skew;
  Matrix2s oRf;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-2d.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ContactModel2DTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ContactData2DTpl)

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_2D_HPP_
