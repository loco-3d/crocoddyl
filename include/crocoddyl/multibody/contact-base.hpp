///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/multibody/force-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

class ContactModelBase {
 public:
  virtual ~ContactModelBase() = default;

  CROCODDYL_BASE_CAST(ContactModelBase, ContactModelAbstractTpl)
};

template <typename _Scalar>
class ContactModelAbstractTpl : public ContactModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact abstraction
   *
   * @param[in] state  State of the multibody system
   * @param[in] type   Type of contact
   * @param[in] nc     Dimension of the contact model
   * @param[in] nu     Dimension of the control vector
   */
  ContactModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                          const pinocchio::ReferenceFrame type,
                          const std::size_t nc, const std::size_t nu);
  ContactModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                          const pinocchio::ReferenceFrame type,
                          const std::size_t nc);

  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ContactModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                              const std::size_t nc, const std::size_t nu);)
  DEPRECATED(
      "Use constructor that passes the type type of contact, this assumes is "
      "pinocchio::LOCAL",
      ContactModelAbstractTpl(std::shared_ptr<StateMultibody> state,
                              const std::size_t nc);)
  virtual ~ContactModelAbstractTpl() = default;

  /**
   * @brief Compute the contact Jacobian and acceleration drift
   *
   * @param[in] data  Contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) = 0;

  /**
   * @brief Compute the derivatives of the acceleration-based contact
   *
   * @param[in] data  Contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ContactDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) = 0;

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   Contact data
   * @param[in] force  Contact force
   */
  virtual void updateForce(const std::shared_ptr<ContactDataAbstract>& data,
                           const VectorXs& force) = 0;

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   Contact data
   * @param[in] force  Contact force
   */
  void updateForceDiff(const std::shared_ptr<ContactDataAbstract>& data,
                       const MatrixXs& df_dx, const MatrixXs& df_du) const;

  /**
   * @brief Set the stack of spatial forces to zero
   *
   * @param[in] data  Contact data
   */
  void setZeroForce(const std::shared_ptr<ContactDataAbstract>& data) const;

  /**
   * @brief Set the stack of spatial forces Jacobians to zero
   *
   * @param[in] data  Contact data
   */
  void setZeroForceDiff(const std::shared_ptr<ContactDataAbstract>& data) const;

  /**
   * @brief Create the contact data
   */
  virtual std::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateMultibody>& get_state() const;

  /**
   * @brief Return the dimension of the contact
   */
  std::size_t get_nc() const;

  /**
   * @brief Return the dimension of the control vector
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the type of contact
   */
  void set_type(const pinocchio::ReferenceFrame type);

  /**
   * @brief Return the type of contact
   */
  pinocchio::ReferenceFrame get_type() const;

  /**
   * @brief Print information on the contact model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ContactModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::shared_ptr<StateMultibody> state_;
  std::size_t nc_;
  std::size_t nu_;
  pinocchio::FrameIndex id_;        //!< Reference frame id of the contact
  pinocchio::ReferenceFrame type_;  //!< Type of contact
  ContactModelAbstractTpl() : state_(nullptr), nc_(0), nu_(0), id_(0) {};
};

template <typename _Scalar>
struct ContactDataAbstractTpl : public ForceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ForceDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;

  template <template <typename Scalar> class Model>
  ContactDataAbstractTpl(Model<Scalar>* const model,
                         pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        fXj(jMf.inverse().toActionMatrix()),
        a0(model->get_nc()),
        da0_dx(model->get_nc(), model->get_state()->get_ndx()),
        dtau_dq(model->get_state()->get_nv(), model->get_state()->get_nv()) {
    a0.setZero();
    da0_dx.setZero();
    dtau_dq.setZero();
  }
  virtual ~ContactDataAbstractTpl() = default;

  using Base::df_du;
  using Base::df_dx;
  using Base::f;
  using Base::frame;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;

  typename SE3::ActionMatrixType fXj;
  VectorXs a0;
  MatrixXs da0_dx;
  MatrixXs dtau_dq;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contact-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ContactModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ContactDataAbstractTpl)

#endif  // CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
