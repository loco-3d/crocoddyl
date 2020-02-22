///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"

#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename _Scalar>
class ContactModel3DTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef ContactData3DTpl<Scalar> ContactData3D;
  typedef FrameTranslationTpl<Scalar> FrameTranslation;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ContactModel3DTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref, const std::size_t& nu,
                    const Vector2s& gains = Vector2s::Zero());
  ContactModel3DTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                    const Vector2s& gains = Vector2s::Zero());
  ~ContactModel3DTpl();

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  const FrameTranslation& get_xref() const;
  const Vector2s& get_gains() const;

 protected:
  using Base::nc_;
  using Base::nu_;
  using Base::state_;

 private:
  FrameTranslation xref_;
  Vector2s gains_;
};

template <typename _Scalar>
struct ContactData3DTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ContactData3DTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fXjdv_dq(6, model->get_state()->get_nv()),
        fXjda_dq(6, model->get_state()->get_nv()),
        fXjda_dv(6, model->get_state()->get_nv()) {
    frame = model->get_xref().frame;
    joint = model->get_state()->get_pinocchio().frames[frame].parent;
    jMf = model->get_state()->get_pinocchio().frames[frame].placement;
    fXj = jMf.inverse().toActionMatrix();
    fJf.fill(0);
    v_partial_dq.fill(0);
    a_partial_dq.fill(0);
    a_partial_dv.fill(0);
    a_partial_da.fill(0);
    fXjdv_dq.fill(0);
    fXjda_dq.fill(0);
    fXjda_dv.fill(0);
    vv.fill(0);
    vw.fill(0);
    vv_skew.fill(0);
    vw_skew.fill(0);
    oRf.fill(0);
  }

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
  Matrix3s oRf;

  using Base::a0;
  using Base::da0_dx;
  using Base::df_du;
  using Base::df_dx;
  using Base::f;
  using Base::frame;
  using Base::fXj;
  using Base::Jc;
  using Base::jMf;
  using Base::joint;
  using Base::pinocchio;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-3d.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
