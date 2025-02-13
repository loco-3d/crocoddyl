///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
#define CROCODDYL_MULTIBODY_FORCE_BASE_HPP_

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/force.hpp>

#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ForceDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::SE3Tpl<Scalar>::ActionMatrixType SE3ActionMatrix;
  typedef pinocchio::MotionTpl<Scalar> Motion;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ForceDataAbstractTpl(Model<Scalar>* const model)
      : frame(0),
        type(model->get_type()),
        jMf(SE3::Identity()),
        fXj(jMf.inverse().toActionMatrix()),
        f(Force::Zero()),
        fext(Force::Zero()),
        oMf(SE3::Identity()),
        fvf(Motion::Zero()),
        faf(Motion::Zero()),
        v_partial_dq(6, model->get_state()->get_nv()),
        v_partial_dv(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()),
        f_v_partial_dq(6, model->get_state()->get_nv()),
        f_a_partial_dq(6, model->get_state()->get_nv()),
        f_a_partial_dv(6, model->get_state()->get_nv()),
        jJj(6, model->get_state()->get_nv()),
        fJf(6, model->get_state()->get_nv()) {
    v_partial_dq.setZero();
    v_partial_dv.setZero();
    a_partial_dq.setZero();
    a_partial_dv.setZero();
    a_partial_da.setZero();
    f_v_partial_dq.setZero();
    f_a_partial_dq.setZero();
    f_a_partial_dv.setZero();
    jJj.setZero();
    fJf.setZero();
  }

  pinocchio::FrameIndex frame;     //!< Frame index of the contact frame
  pinocchio::ReferenceFrame type;  //!< Type of contact
  SE3 jMf;  //!< Local frame placement of the contact frame
  typename SE3::ActionMatrixType fXj;  //<! Action matrix that transforms the
                                       // contact force to the joint torques
  Force f;     //!< Contact force expressed in the coordinate defined by type
  Force fext;  //!< External spatial force at the parent joint level
  SE3 oMf;     //<! Placement in the world frame
  Motion fvf;  //<! Frame velocity
  Motion faf;  //<! Frame acceleration

  Matrix6xs v_partial_dq;  //!< Partial derivative of velcoity w.r.t. the joint
                           //!< configuration
  Matrix6xs v_partial_dv;  //!< Partial derivative of velcoity w.r.t. the joint
                           //!< velocity
  Matrix6xs a_partial_dq;  //!< Partial derivative of acceleration w.r.t. the
                           //!< joint configuration
  Matrix6xs a_partial_dv;  //!< Partial derivative of acceleration w.r.t. the
                           //!< joint velocity
  Matrix6xs a_partial_da;  //!< Partial derivative of acceleration w.r.t. the
                           //!< joint acceleration
  Matrix6xs f_v_partial_dq;  //!< Partial derivative of velocity w.r.t. the
                             //!< joint configuration in local frame
  Matrix6xs f_a_partial_dq;  //!< Partial derivative of acceleration w.r.t. the
                             //!< joint configuration in local frame
  Matrix6xs f_a_partial_dv;  //!< Partial derivative of acceleration w.r.t. the
                             //!< joint velocity in local frame
  MatrixXs jJj;              //<! Joint jacobian
  MatrixXs fJf;              //<! Frame jacobian
};

template <typename _Scalar>
struct InteractionDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  InteractionDataAbstractTpl(Model<Scalar>* const model, const int nf)
      : nf(nf),
        force_datas(nf, ForceDataAbstractTpl<Scalar>(model)),
        df_dx(model->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_nc(), model->get_nu()) {
    df_dx.setZero();
    df_du.setZero();
  }

  virtual ~InteractionDataAbstractTpl() {}

  std::size_t nf;
  std::vector<ForceDataAbstractTpl<Scalar>> force_datas;

  MatrixXs df_dx;  //!< Jacobian of the contact forces expressed in the
                   //!< coordinate defined by type
  MatrixXs df_du;  //!< Jacobian of the contact forces expressed in the
                   //!< coordinate defined by type
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
