///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_1D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_1D_HPP_

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModel1DTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactData1DTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the 1d contact model
   *
   * To learn more about the computation of the contact derivatives in different
   * frames see
   *  S. Kleff et. al, On the Derivation of the Contact Dynamics in Arbitrary
   *  Frames: Application to Polishing with Talos, ICHR 2022
   *
   * @param[in] state     State of the multibody system
   * @param[in] id        Reference frame id of the contact
   * @param[in] xref      Contact position used for the Baumgarte stabilization
   * @param[in] type      Type of contact
   * @param[in] rotation  Rotation of the reference frame's z-axis
   * @param[in] nu        Dimension of the control vector
   * @param[in] gains     Baumgarte stabilization gains
   */
  ContactModel1DTpl(boost::shared_ptr<StateMultibody> state,
                    const pinocchio::FrameIndex id, const Scalar xref,
                    const pinocchio::ReferenceFrame type,
                    const Matrix3s& rotation, const std::size_t nu,
                    const Vector2s& gains = Vector2s::Zero());

  /**
   * @brief Initialize the 1d contact model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`. To learn
   * more about the computation of the contact derivatives in different frames
   * see
   *  S. Kleff et. al, On the Derivation of the Contact Dynamics in Arbitrary
   *  Frames: Application to Polishing with Talos, ICHR 2022
   *
   * @param[in] state     State of the multibody system
   * @param[in] id        Reference frame id of the contact
   * @param[in] xref      Contact position used for the Baumgarte stabilization
   * @param[in] type      Type of contact
   * @param[in] gains     Baumgarte stabilization gains
   */
  ContactModel1DTpl(boost::shared_ptr<StateMultibody> state,
                    const pinocchio::FrameIndex id, const Scalar xref,
                    const pinocchio::ReferenceFrame type,
                    const Vector2s& gains = Vector2s::Zero());

  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ContactModel1DTpl(boost::shared_ptr<StateMultibody> state,
                        const pinocchio::FrameIndex id, const Scalar xref,
                        const std::size_t nu,
                        const Vector2s& gains = Vector2s::Zero());)
  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ContactModel1DTpl(boost::shared_ptr<StateMultibody> state,
                        const pinocchio::FrameIndex id, const Scalar xref,
                        const Vector2s& gains = Vector2s::Zero());)
  virtual ~ContactModel1DTpl();

  /**
   * @brief Compute the 1d contact Jacobian and drift
   *
   * @param[in] data  1d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the 1d contact holonomic constraint
   *
   * @param[in] data  1d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   1d contact data
   * @param[in] force  1d force
   */
  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                           const VectorXs& force);

  /**
   * @brief Create the 1d contact data
   */
  virtual boost::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Return the reference frame translation
   */
  const Scalar get_reference() const;

  /**
   * @brief Create the 1d contact data
   */
  const Vector2s& get_gains() const;

  /**
   * @brief Return the rotation of the reference frames's z axis
   */
  const Matrix3s& get_axis_rotation() const;

  /**
   * @brief Modify the reference frame translation
   */
  void set_reference(const Scalar reference);

  /**
   * @brief Modify the rotation of the reference frames's z axis
   */
  void set_axis_rotation(const Matrix3s& rotation);

  /**
   * @brief Print relevant information of the 1d contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::id_;
  using Base::nc_;
  using Base::nu_;
  using Base::state_;
  using Base::type_;

 private:
  Scalar xref_;     //!< Contact position used for the Baumgarte stabilization
  Matrix3s Raxis_;  //!< Rotation of the reference frame's z-axis
  Vector2s gains_;  //!< Baumgarte stabilization gains
};

template <typename _Scalar>
struct ContactData1DTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Matrix2s Matrix2s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename pinocchio::MotionTpl<Scalar> Motion;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ContactData1DTpl(Model<Scalar>* const model,
                   pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data, 1),
        v(Motion::Zero()),
        f_local(Force::Zero()),
        da0_local_dx(3, model->get_state()->get_ndx()),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fXjdv_dq(6, model->get_state()->get_nv()),
        fXjda_dq(6, model->get_state()->get_nv()),
        fXjda_dv(6, model->get_state()->get_nv()),
        fJf_df(3, model->get_state()->get_nv()) {
    // There is only one element in the force_datas vector
    ForceDataAbstract& fdata = force_datas[0];
    fdata.frame = model->get_id();
    fdata.jMf = model->get_state()->get_pinocchio()->frames[fdata.frame].placement;
    fdata.fXj = fdata.jMf.inverse().toActionMatrix();

    a0_local.setZero();
    dp.setZero();
    dp_local.setZero();
    da0_local_dx.setZero();
    fJf.setZero();
    v_partial_dq.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    vv_skew.setZero();
    vw_skew.setZero();
    a0_skew.setZero();
    a0_world_skew.setZero();
    dp_skew.setZero();
    f_skew.setZero();
    fXjdv_dq.setZero();
    fXjda_dq.setZero();
    fXjda_dv.setZero();
    fJf_df.setZero();
  }

  using Base::a0;
  using Base::da0_dx;
  using Base::Jc;
  using Base::pinocchio;
  using Base::force_datas;

  Motion v;
  Vector3s a0_local;
  Vector3s dp;
  Vector3s dp_local;
  Force f_local;
  Matrix3xs da0_local_dx;
  Matrix6xs fJf;
  Matrix6xs v_partial_dq;
  Matrix6xs a_partial_dq;
  Matrix6xs a_partial_dv;
  Matrix6xs a_partial_da;
  Matrix3s vv_skew;
  Matrix3s vw_skew;
  Matrix3s a0_skew;
  Matrix3s a0_world_skew;
  Matrix3s dp_skew;
  Matrix3s f_skew;
  Matrix6xs fXjdv_dq;
  Matrix6xs fXjda_dq;
  Matrix6xs fXjda_dv;
  Matrix3xs fJf_df;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-1d.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_1D_HPP_
