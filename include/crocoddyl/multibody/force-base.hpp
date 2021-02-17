///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
#define CROCODDYL_MULTIBODY_FORCE_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

template <typename _Scalar>
struct ForceDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ForceDataAbstractTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : pinocchio(data),
        joint(0),
        frame(0),
        jMf(pinocchio::SE3Tpl<Scalar>::Identity()),
        Jc(model->get_nc(), model->get_state()->get_nv()),
        f(pinocchio::ForceTpl<Scalar>::Zero()),
        df_dx(model->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_nc(), model->get_nu()) {
    Jc.setZero();
    df_dx.setZero();
    df_du.setZero();
  }
  virtual ~ForceDataAbstractTpl() {}

  typename pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  pinocchio::JointIndex joint;                     //!< Joint index of the contact frame
  pinocchio::FrameIndex frame;                     //!< Frame index of the contact frame
  typename pinocchio::SE3Tpl<Scalar> jMf;          //!< Local frame placement of the contact frame
  MatrixXs Jc;                                     //!< Contact Jacobian
  pinocchio::ForceTpl<Scalar> f;  //!< External spatial force at the parent joint level. Note that we could compute the
                                  //!< force at the contact frame by using jMf (i.e. data.jMf.actInv(data.f)
  MatrixXs df_dx;                 //!< Jacobian of the contact forces
  MatrixXs df_du;                 //!< Jacobian of the contact forces
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
