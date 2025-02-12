///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModel6DTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactData6DTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the 6d contact model
   *
   * To learn more about the computation of the contact derivatives in different
   * frames see
   *  S. Kleff et. al, On the Derivation of the Contact Dynamics in Arbitrary
   *  Frames: Application to Polishing with Talos, ICHR 2022
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id of the contact
   * @param[in] pref   Contact placement used for the Baumgarte stabilization
   * @param[in] type   Type of contact
   * @param[in] nu     Dimension of the control vector
   * @param[in] gains  Baumgarte stabilization gains
   */
  ContactModel6DTpl(boost::shared_ptr<StateMultibody> state,
                    const pinocchio::FrameIndex id, const SE3& pref,
                    const pinocchio::ReferenceFrame type, const std::size_t nu,
                    const Vector2s& gains = Vector2s::Zero());

  /**
   * @brief Initialize the 6d contact model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`. To learn
   * more about the computation of the contact derivatives in different frames
   * see
   *  S. Kleff et. al, On the Derivation of the Contact Dynamics in Arbitrary
   *  Frames: Application to Polishing with Talos, ICHR 2022
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id of the contact
   * @param[in] pref   Contact placement used for the Baumgarte stabilization
   * @param[in] type   Type of contact
   * @param[in] gains  Baumgarte stabilization gains
   */
  ContactModel6DTpl(boost::shared_ptr<StateMultibody> state,
                    const pinocchio::FrameIndex id, const SE3& pref,
                    const pinocchio::ReferenceFrame type,
                    const Vector2s& gains = Vector2s::Zero());

  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ContactModel6DTpl(boost::shared_ptr<StateMultibody> state,
                        const pinocchio::FrameIndex id, const SE3& pref,
                        const std::size_t nu,
                        const Vector2s& gains = Vector2s::Zero());)
  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ContactModel6DTpl(boost::shared_ptr<StateMultibody> state,
                        const pinocchio::FrameIndex id, const SE3& pref,
                        const Vector2s& gains = Vector2s::Zero());)
  virtual ~ContactModel6DTpl();

  /**
   * @brief Compute the 3d contact Jacobian and drift
   *
   * @param[in] data  3d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the 6d contact holonomic constraint
   *
   * @param[in] data  6d contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   6d contact data
   * @param[in] force  6d force
   */
  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract>& data,
                           const VectorXs& force);

  /**
   * @brief Create the 6d contact data
   */
  virtual boost::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Return the reference frame placement
   */
  const SE3& get_reference() const;

  /**
   * @brief Return the Baumgarte stabilization gains
   */
  const Vector2s& get_gains() const;

  /**
   * @brief Modify the reference frame placement
   */
  void set_reference(const SE3& reference);

  /**
   * @brief Print relevant information of the 6d contact model
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
  SE3 pref_;        //!< Contact placement used for the Baumgarte stabilization
  Vector2s gains_;  //!< Baumgarte stabilization gains
};

template <typename _Scalar>
struct ContactData6DTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::MotionTpl<Scalar> Motion;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ContactData6DTpl(Model<Scalar>* const model,
                   pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data, 1),
        rMf(SE3::Identity()),
        lwaMl(SE3::Identity()),
        v(Motion::Zero()),
        a0_local(Motion::Zero()),
        f_local(Force::Zero()),
        da0_local_dx(6, model->get_state()->get_ndx()),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        fJf_df(6, model->get_state()->get_nv()) {
    // There is only one element in the force_datas vector
    ForceDataAbstract& fdata = force_datas[0];
    fdata.frame = model->get_id();
    fdata.jMf = model->get_state()->get_pinocchio()->frames[fdata.frame].placement;
    fdata.fXj = fdata.jMf.inverse().toActionMatrix();

    da0_local_dx.setZero();
    fJf.setZero();
    v_partial_dq.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    av_world_skew.setZero();
    aw_world_skew.setZero();
    av_skew.setZero();
    aw_skew.setZero();
    fv_skew.setZero();
    fw_skew.setZero();
    rMf_Jlog6.setZero();
    fJf_df.setZero();
  }

  using Base::a0;
  using Base::da0_dx;
  using Base::Jc;
  using Base::pinocchio;
  using Base::force_datas;

  SE3 rMf;
  SE3 lwaMl;
  Motion v;
  Motion a0_local;
  Force f_local;
  Matrix6xs da0_local_dx;
  Matrix6xs fJf;
  Matrix6xs v_partial_dq;
  Matrix6xs a_partial_dq;
  Matrix6xs a_partial_dv;
  Matrix6xs a_partial_da;
  Matrix3s av_world_skew;
  Matrix3s aw_world_skew;
  Matrix3s av_skew;
  Matrix3s aw_skew;
  Matrix3s fv_skew;
  Matrix3s fw_skew;
  Matrix6s rMf_Jlog6;
  MatrixXs fJf_df;
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/contact-6d.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
