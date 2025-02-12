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

  typedef typename pinocchio::SE3Tpl<Scalar> SE3;
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ForceDataAbstractTpl(Model<Scalar>* const model)
      : frame(0),
        type(model->get_type()),
        jMf(SE3::Identity()),
        fXj(jMf.inverse().toActionMatrix()),
        f(Force::Zero()),
        fext(Force::Zero()) {}
  virtual ~ForceDataAbstractTpl() {}

  pinocchio::FrameIndex frame;     //!< Frame index of the contact frame
  pinocchio::ReferenceFrame type;  //!< Type of contact
  SE3 jMf;  //!< Local frame placement of the contact frame
  typename SE3::ActionMatrixType fXj;  //<! Action matrix that transforms the
                                       // contact force to the joint torques
  Force f;     //!< Contact force expressed in the coordinate defined by type
  Force fext;  //!< External spatial force at the parent joint level
};

template <typename _Scalar>
struct InteractionDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  InteractionDataAbstractTpl(Model<Scalar>* const model,
                             pinocchio::DataTpl<Scalar>* const data,
                             const int nf)
      : nf(nf),
        force_datas(nf, ForceDataAbstractTpl<Scalar>(model)),
        df_dx(model->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_nc(), model->get_nu()) {
    df_dx.setZero();
    df_du.setZero();
  }

  virtual ~InteractionDataAbstractTpl() {}

  std::size_t nf;
  std::vector<ForceDataAbstract> force_datas;

  MatrixXs df_dx;  //!< Jacobian of the contact forces expressed in the
                   //!< coordinate defined by type
  MatrixXs df_du;  //!< Jacobian of the contact forces expressed in the
                   //!< coordinate defined by type
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FORCE_BASE_HPP_
