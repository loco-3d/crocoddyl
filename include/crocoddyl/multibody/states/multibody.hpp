///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
#define CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include <pinocchio/multibody/model.hpp>

namespace crocoddyl {

template <typename _Scalar>
class StateMultibodyTpl : public StateAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  enum JointType { FreeFlyer = 0, Spherical, Simple };

  explicit StateMultibodyTpl(boost::shared_ptr<pinocchio::ModelTpl<Scalar> > model);
  virtual ~StateMultibodyTpl();

  virtual VectorXs zero() const;
  virtual VectorXs rand() const;
  virtual void diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const;
  virtual void integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const;
  virtual void Jdiff(const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&, Eigen::Ref<MatrixXs> Jfirst,
                     Eigen::Ref<MatrixXs> Jsecond, Jcomponent firstsecond = both) const;
  virtual void Jintegrate(const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&,
                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                          Jcomponent firstsecond = both) const;

  const boost::shared_ptr<pinocchio::ModelTpl<Scalar> >& get_pinocchio() const;

 protected:
  using Base::has_limits_;
  using Base::lb_;
  using Base::ndx_;
  using Base::nq_;
  using Base::nv_;
  using Base::nx_;
  using Base::ub_;

 private:
  
  boost::shared_ptr<pinocchio::ModelTpl<Scalar> > pinocchio_;
  VectorXs x0_;
  JointType joint_type_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/states/multibody.hxx"

#endif  // CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
