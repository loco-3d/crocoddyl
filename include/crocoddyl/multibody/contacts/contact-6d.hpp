///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/frames.hpp"

#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

template <typename _Scalar>
class ContactModel6DTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef ContactData6DTpl<Scalar> ContactData6D;
  typedef FramePlacementTpl<Scalar> FramePlacement;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ContactModel6DTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& xref, const std::size_t& nu,
                    const Vector2s& gains = Vector2s::Zero());
  ContactModel6DTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& xref,
                    const Vector2s& gains = Vector2s::Zero());
  ~ContactModel6DTpl();

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  const FramePlacement& get_Mref() const;
  const Vector2s& get_gains() const;

 protected:
  using Base::nc_;
  using Base::nu_;
  using Base::state_;

 private:
  FramePlacement Mref_;
  Vector2s gains_;
};

template <typename _Scalar>
struct ContactData6DTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;

  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ContactData6DTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        rMf(pinocchio::SE3Tpl<Scalar>::Identity()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()) {
    frame = model->get_Mref().frame;
    joint = model->get_state()->get_pinocchio()->frames[frame].parent;
    jMf = model->get_state()->get_pinocchio()->frames[frame].placement;
    fXj = jMf.inverse().toActionMatrix();
    v_partial_dq.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    rMf_Jlog6.setZero();
  }

  using Base::pinocchio;
  using Base::joint;
  using Base::frame;
  using Base::jMf;
  using Base::fXj;
  using Base::Jc;
  using Base::a0;
  using Base::da0_dx;
  using Base::f;
  using Base::df_du;
  using Base::df_dx;

  pinocchio::SE3Tpl<Scalar> rMf;
  pinocchio::MotionTpl<Scalar> v;
  pinocchio::MotionTpl<Scalar> a;
  Matrix6xs v_partial_dq;
  Matrix6xs a_partial_dq;
  Matrix6xs a_partial_dv;
  Matrix6xs a_partial_da;
  Matrix6s rMf_Jlog6;
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-6d.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
