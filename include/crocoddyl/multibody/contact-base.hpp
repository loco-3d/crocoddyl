///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

template <typename _Scalar>
class ContactModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nc, const std::size_t& nu);
  ContactModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nc);
  virtual ~ContactModelAbstractTpl();

  virtual void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) = 0;

  virtual void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force) = 0;
  void updateForceDiff(const boost::shared_ptr<ContactDataAbstract>& data, const MatrixXs& df_dx,
                       const MatrixXs& df_du) const;
  virtual boost::shared_ptr<ContactDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const std::size_t& get_nc() const;
  const std::size_t& get_nu() const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  std::size_t nc_;
  std::size_t nu_;
};

template <typename _Scalar>
struct ContactDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ContactDataAbstractTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : pinocchio(data),
        joint(0),
        frame(0),
        jMf(pinocchio::SE3Tpl<Scalar>::Identity()),
        fXj(jMf.inverse().toActionMatrix()),
        Jc(model->get_nc(), model->get_state()->get_nv()),
        a0(model->get_nc()),
        da0_dx(model->get_nc(), model->get_state()->get_ndx()),
        f(pinocchio::ForceTpl<Scalar>::Zero()),
        df_dx(model->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_nc(), model->get_nu()) {
    Jc.setZero();
    a0.setZero();
    da0_dx.setZero();
    df_dx.setZero();
    df_du.setZero();
  }
  virtual ~ContactDataAbstractTpl() {}

  typename pinocchio::DataTpl<Scalar>* pinocchio;
  pinocchio::JointIndex joint;
  pinocchio::FrameIndex frame;
  typename pinocchio::SE3Tpl<Scalar> jMf;
  typename pinocchio::SE3Tpl<Scalar>::ActionMatrixType fXj;
  MatrixXs Jc;
  VectorXs a0;
  MatrixXs da0_dx;
  pinocchio::ForceTpl<Scalar> f;
  MatrixXs df_dx;
  MatrixXs df_du;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contact-base.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
