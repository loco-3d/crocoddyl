///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
#define CROCODDYL_MULTIBODY_FORCE_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ForceDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename pinocchio::DataTpl<Scalar> PinocchioData;
  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ForceDataAbstractTpl(Model<Scalar>* const model, PinocchioData* const data)
      : pinocchio(data),
        frame(0),
        type(model->get_type()),
        jMf(SE3::Identity()),
        Jc(model->get_nc(), model->get_state()->get_nv()),
        f(Force::Zero()),
        fext(Force::Zero()),
        df_dx(model->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_nc(), model->get_nu()) {
    Jc.setZero();
    df_dx.setZero();
    df_du.setZero();
  }
  virtual ~ForceDataAbstractTpl() = default;

  PinocchioData* pinocchio;        //!< Pinocchio data
  pinocchio::FrameIndex frame;     //!< Frame index of the contact frame
  pinocchio::ReferenceFrame type;  //!< Type of contact
  SE3 jMf;      //!< Local frame placement of the contact frame
  MatrixXs Jc;  //!< Contact Jacobian
  Force f;      //!< Contact force expressed in the coordinate defined by type
  Force fext;   //!< External spatial force at the parent joint level
  MatrixXs df_dx;  //!< Jacobian of the contact forces expressed in the
                   //!< coordinate defined by type
  MatrixXs df_du;  //!< Jacobian of the contact forces expressed in the
                   //!< coordinate defined by type
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ForceDataAbstractTpl)

#endif  // CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
