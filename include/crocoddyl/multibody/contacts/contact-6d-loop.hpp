///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_LOOP_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_LOOP_HPP_

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModel6DLoopTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactData6DLoopTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6s Matrix6s;

  /**
   * @brief Initialize the 6d contact model from joint and placements
   *
   *
   * @param[in] state  State of the multibody system
   * @param[in] joint1_id     Reference joint id of the first contact
   * @param[in] joint1_placement     Placement of the first contact with respect
   * to the joint
   * @param[in] joint2_id     Reference joint id of the second contact
   * @param[in] joint2_placement     Placement of the second contact with
   * respect to the joint
   * @param[in] ref    Reference frame of contact
   * @param[in] nu     Dimension of the control vector
   * @param[in] gains  Baumgarte stabilization gains
   */
  ContactModel6DLoopTpl(boost::shared_ptr<StateMultibody> state,
                        const int joint1_id, const SE3 &joint1_placement,
                        const int joint2_id, const SE3 &joint2_placement,
                        const pinocchio::ReferenceFrame ref,
                        const std::size_t nu,
                        const Vector2s &gains = Vector2s());

  /**
   * @brief Initialize the 6d contact model from joint and placements
   *
   *
   * @param[in] state  State of the multibody system
   * @param[in] joint1_id     Reference joint id of the first contact
   * @param[in] joint1_placement     Placement of the first contact with respect
   * to the joint
   * @param[in] joint2_id     Reference joint id of the second contact
   * @param[in] joint2_placement     Placement of the second contact with
   * respect to the joint
   * @param[in] ref    Reference frame of contact
   * @param[in] gains  Baumgarte stabilization gains
   */
  ContactModel6DLoopTpl(boost::shared_ptr<StateMultibody> state,
                        const int joint1_id, const SE3 &joint1_placement,
                        const int joint2_id, const SE3 &joint2_placement,
                        const pinocchio::ReferenceFrame ref,
                        const Vector2s &gains = Vector2s::Zero());

  virtual ~ContactModel6DLoopTpl();

  /**
   * @brief Compute the 3d contact Jacobian and drift
   *
   * @param[in] data  3d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ContactDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);

  /**
   * @brief Compute the derivatives of the 6d contact holonomic constraint
   *
   * @param[in] data  6d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   6d contact data
   * @param[in] force  6d force
   */
  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract> &data,
                           const VectorXs &force);

  /**
   * @brief Create the 6d contact data
   */
  virtual boost::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar> *const data);

  /**
   * @brief Return the reference the first contact frame parent joint
   */
  const int get_joint1_id() const;

  /**
   * @brief Return the reference the first contact frame placement
   */
  const SE3 &get_joint1_placement() const;

  /**
   * @brief Return the reference the second contact frame parent joint
   */
  const int get_joint2_id() const;

  /**
   * @brief Return the reference the second contact frame placement
   */
  const SE3 &get_joint2_placement() const;

  /**
   * @brief Return the Baumgarte stabilization gains
   */
  const Vector2s &get_gains() const;

  /**
   * @brief Print relevant information of the 6d contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

 protected:
  using Base::id_;
  using Base::nc_;
  using Base::nu_;
  using Base::state_;
  using Base::type_;

 private:
  int joint1_id_;         //!< Reference joint id of the first contact
  SE3 joint1_placement_;  //!< Placement of the first contact with respect to
                          //!< the joint
  int joint2_id_;         //!< Reference joint id of the second contact
  SE3 joint2_placement_;  //!< Placement of the second contact with respect to
                          //!< the joint
  Vector2s gains_;  //!< Baumgarte stabilization gains
};

template <typename _Scalar>
struct ContactData6DLoopTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::SE3Tpl<Scalar>::ActionMatrixType SE3ActionMatrix;
  typedef typename pinocchio::MotionTpl<Scalar> Motion;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ContactData6DLoopTpl(Model<Scalar> *const model,
                       pinocchio::DataTpl<Scalar> *const data)
      : Base(model, data),
        v1_partial_dq(6, model->get_state()->get_nv()),
        a1_partial_dq(6, model->get_state()->get_nv()),
        a1_partial_dv(6, model->get_state()->get_nv()),
        a1_partial_da(6, model->get_state()->get_nv()),
        v2_partial_dq(6, model->get_state()->get_nv()),
        a2_partial_dq(6, model->get_state()->get_nv()),
        a2_partial_dv(6, model->get_state()->get_nv()),
        a2_partial_da(6, model->get_state()->get_nv()),
        da0_dq_t1(6, model->get_state()->get_nv()),
        da0_dq_t2(6, model->get_state()->get_nv()),
        da0_dq_t3(6, model->get_state()->get_nv()) {
    v1_partial_dq.setZero();
    a1_partial_dq.setZero();
    a1_partial_dv.setZero();
    a1_partial_da.setZero();
    v2_partial_dq.setZero();
    a2_partial_dq.setZero();
    a2_partial_dv.setZero();
    a2_partial_da.setZero();
    da0_dq_t1.setZero();
    da0_dq_t2.setZero();
    da0_dq_t3.setZero();
    //
    j1Xf1 = SE3ActionMatrix::Identity();
    j2Xf2 = SE3ActionMatrix::Identity();
    f1Mf2 = SE3::Identity();
    f1Xf2 = SE3ActionMatrix::Identity();
    //
    f1Jf1 = MatrixXs::Zero(6, model->get_state()->get_nv());
    f2Jf2 = MatrixXs::Zero(6, model->get_state()->get_nv());
    j1Jj1 = MatrixXs::Zero(6, model->get_state()->get_nv());
    j2Jj2 = MatrixXs::Zero(6, model->get_state()->get_nv());
    //
    f1vf1 = Motion::Zero();
    f2vf2 = Motion::Zero();
    f1vf2 = Motion::Zero();
    //
    f1af1 = Motion::Zero();
    f2af2 = Motion::Zero();
    f1af2 = Motion::Zero();
    //
    joint1_f = Force::Zero();
    joint2_f = Force::Zero();
  }

  using Base::a0;
  using Base::da0_dx;
  using Base::df_du;
  using Base::df_dx;
  using Base::f;
  using Base::Jc;
  using Base::pinocchio;

  Matrix6xs v1_partial_dq;
  Matrix6xs a1_partial_dq;
  Matrix6xs a1_partial_dv;
  Matrix6xs a1_partial_da;
  Matrix6xs v2_partial_dq;
  Matrix6xs a2_partial_dq;
  Matrix6xs a2_partial_dv;
  Matrix6xs a2_partial_da;

  Matrix6xs da0_dq_t1;
  Matrix6xs da0_dq_t2;
  Matrix6xs da0_dq_t3;

  // Placement related data
  SE3 oMf1;   // Placement of the first contact frame in the world frame
  SE3 oMf2;   // Placement of the second contact frame in the world frame
  SE3 f1Mf2;  // Relative placement of the contact frames in f1 frame
  SE3ActionMatrix j1Xf1;
  SE3ActionMatrix j2Xf2;
  SE3ActionMatrix f1Xf2;
  // Jacobian related data
  MatrixXs f1Jf1;
  MatrixXs f2Jf2;
  MatrixXs j1Jj1;
  MatrixXs j2Jj2;
  // Velocity related data
  Motion f1vf1;
  Motion f2vf2;
  Motion f1vf2;
  // Acceleration related data
  Motion f1af1;
  Motion f2af2;
  Motion f1af2;
  // Force related data
  Force joint1_f;
  Force joint2_f;
  Force f_local;

  Matrix6s f_cross;
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-6d-loop.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_LOOP_HPP_
