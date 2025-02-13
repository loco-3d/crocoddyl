///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
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
  typedef ContactData6DLoopTpl<Scalar> Data;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the 6d loop-contact model from joint and placements
   *
   *
   * @param[in] state             State of the multibody system
   * @param[in] joint1_id         Parent joint id of the first contact
   * @param[in] joint1_placement  Placement of the first contact with
   *                                  respect to the parent joint
   * @param[in] joint2_id         Parent joint id of the second contact
   * @param[in] joint2_placement  Placement of the second contact with
   *                                  respect to the parent joint
   * @param[in] type              Reference frame of contact
   * @param[in] nu                Dimension of the control vector
   * @param[in] gains             Baumgarte stabilization gains
   */
  ContactModel6DLoopTpl(boost::shared_ptr<StateMultibody> state,
                        const int joint1_id, const SE3 &joint1_placement,
                        const int joint2_id, const SE3 &joint2_placement,
                        const pinocchio::ReferenceFrame type,
                        const std::size_t nu,
                        const Vector2s &gains = Vector2s::Zero());

  /**
   * @brief Initialize the 6d loop-contact model from joint and placements
   *
   *
   * @param[in] state             State of the multibody system
   * @param[in] joint1_id         Parent joint id of the first contact
   * @param[in] joint1_placement  Placement of the first contact with
   *                                  respect to the parent joint
   * @param[in] joint2_id         Parent joint id of the second contact
   * @param[in] joint2_placement  Placement of the second contact with
   *                                  respect to the parent joint
   * @param[in] type              Reference frame of contact
   * @param[in] gains             Baumgarte stabilization gains
   */
  ContactModel6DLoopTpl(boost::shared_ptr<StateMultibody> state,
                        const int joint1_id, const SE3 &joint1_placement,
                        const int joint2_id, const SE3 &joint2_placement,
                        const pinocchio::ReferenceFrame type,
                        const Vector2s &gains = Vector2s::Zero());

  virtual ~ContactModel6DLoopTpl();

  /**
   * @brief Compute the 6d loop-contact Jacobian and drift
   *
   * @param[in] data  6d loop-contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ContactDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);

  /**
   * @brief Compute the derivatives of the 6d loop-contact holonomic constraint
   *
   * @param[in] data  6d loop-contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   6d loop-contact data
   * @param[in] force  6d force
   */
  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract> &data,
                           const VectorXs &force);

  /**
   * @brief Updates the force differential for the given contact data.
   *
   * This function updates the force differential matrices with respect to the
   * state and control variables.
   *
   * @param[in] data  Shared pointer to the contact data abstract.
   * @param[in] df_dx Matrix representing the differential of the force with
   * respect to the state variables.
   * @param[in] df_du Matrix representing the differential of the force with
   * respect to the control variables.
   */
  virtual void updateForceDiff(
      const boost::shared_ptr<ContactDataAbstract> &data, const MatrixXs &df_dx,
      const MatrixXs &df_du);

  /**
   * @brief Create the 6d loop-contact data
   */
  virtual boost::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar> *const data);

  const SE3& get_placement(const int force_index) const;

  /**
   * @brief Return the Baumgarte stabilization gains
   */
  const Vector2s &get_gains() const;

  /**
   * @brief Set the first contact frame placement with respect to the parent
   * joint
   */
  void set_placement(const int force_index, const SE3& placement);

  /**
   * @brief Set the Baumgarte stabilization gains
   */
  void set_gains(const Vector2s &gains);

  /**
   * @brief Print relevant information of the 6D loop-contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

 protected:
  using Base::id_;
  using Base::nc_;
  using Base::nf_;
  using Base::nu_;
  using Base::placements_;
  using Base::state_;
  using Base::type_;

 private:
  Vector2s gains_;  //!< Baumgarte stabilization gains
};

template <typename _Scalar>
struct ContactData6DLoopTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::SE3Tpl<Scalar>::ActionMatrixType SE3ActionMatrix;
  typedef pinocchio::MotionTpl<Scalar> Motion;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ContactData6DLoopTpl(Model<Scalar> *const model,
                       pinocchio::DataTpl<Scalar> *const data)
      : Base(model, data, 2),
        f1_v2_partial_dq(6, model->get_state()->get_nv()),
        da0_dq_t1(6, model->get_state()->get_nv()),
        da0_dq_t2(6, model->get_state()->get_nv()),
        da0_dq_t2_tmp(6, model->get_state()->get_nv()),
        da0_dq_t3(6, model->get_state()->get_nv()),
        da0_dq_t3_tmp(6, model->get_state()->get_nv()),
        dpos_dq(6, model->get_state()->get_nv()),
        dvel_dq(6, model->get_state()->get_nv()),
        dtau_dq_tmp(model->get_state()->get_nv(), model->get_state()->get_nv()),
        f1Jf2(6, model->get_state()->get_nv()),
        f1Mf2(SE3::Identity()),
        f1Xf2(SE3ActionMatrix::Identity()),
        f1vf2(Motion::Zero()),
        f1af2(Motion::Zero()) {
    if (force_datas.size() != 2 || nf != 2) {
      throw_pretty(
          "Invalid argument: " << "the force_datas has to be of size 2");
    }

    ForceDataAbstract &fdata1 = force_datas[0];
    fdata1.frame = model->get_id(0);
    fdata1.jMf = model->get_placement(0);
    fdata1.fXj = fdata1.jMf.inverse().toActionMatrix();
    fdata1.type = model->get_type();

    ForceDataAbstract &fdata2 = force_datas[1];
    fdata2.frame = model->get_id(1);
    fdata2.jMf = model->get_placement(1);
    fdata2.fXj = fdata2.jMf.inverse().toActionMatrix();
    fdata2.type = model->get_type();

    f1_v2_partial_dq.setZero();
    da0_dq_t1.setZero();
    da0_dq_t2.setZero();
    da0_dq_t2_tmp.setZero();
    da0_dq_t3.setZero();
    da0_dq_t3_tmp.setZero();
    dpos_dq.setZero();
    dvel_dq.setZero();
    dtau_dq_tmp.setZero();
    f1Jf2.setZero();
    j2Jj1.setZero();
  }

  using Base::a0;
  using Base::da0_dx;
  using Base::force_datas;
  using Base::nf;

  // Intermediate calculations
  Matrix6xs da0_dq_t1;
  Matrix6xs da0_dq_t2;
  Matrix6xs da0_dq_t2_tmp;
  Matrix6xs da0_dq_t3;
  Matrix6xs da0_dq_t3_tmp;
  Matrix6xs dpos_dq;
  Matrix6xs dvel_dq;
  MatrixXs dtau_dq_tmp;

  // Coupled terms
  SE3 f1Mf2;  //<! Relative placement of the contact frames in the first contact
              // frame
  SE3ActionMatrix f1Xf2;  //<! Relative action matrix of the
                          // contact frames in the first contact frame
  Motion f1vf2;
  Motion f1af2;
  Matrix6xs f1Jf2;
  Matrix6xs j2Jj1;
  Matrix6xs f1_v2_partial_dq;
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-6d-loop.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_LOOP_HPP_
