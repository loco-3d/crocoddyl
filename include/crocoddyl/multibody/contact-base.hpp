///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/force-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nc, const std::size_t nu);
  ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nc);
  virtual ~ContactModelAbstractTpl();

  virtual void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;

  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) = 0;
  void updateForceDiff(const boost::shared_ptr<ContactDataAbstract>& data, const MatrixXs& df_dx,
                       const MatrixXs& df_du) const;
  void setZeroForce(const boost::shared_ptr<ContactDataAbstract>& data) const;
  void setZeroForceDiff(const boost::shared_ptr<ContactDataAbstract>& data) const;

  virtual boost::shared_ptr<ContactDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  std::size_t get_nc() const;
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
   * @brief Print information on the contact model
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const ContactModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the contact model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  std::size_t nc_;
  std::size_t nu_;
  pinocchio::FrameIndex id_;  //!< Reference frame id of the contact
};

template <typename _Scalar>
struct ContactDataAbstractTpl : public ForceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ForceDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ContactDataAbstractTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        fXj(jMf.inverse().toActionMatrix()),
        a0(model->get_nc()),
        da0_dx(model->get_nc(), model->get_state()->get_ndx()) {
    a0.setZero();
    da0_dx.setZero();
  }
  virtual ~ContactDataAbstractTpl() {}

  using Base::df_du;
  using Base::df_dx;
  using Base::f;
  using Base::frame;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;
  typename pinocchio::SE3Tpl<Scalar>::ActionMatrixType fXj;
  VectorXs a0;
  MatrixXs da0_dx;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contact-base.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
