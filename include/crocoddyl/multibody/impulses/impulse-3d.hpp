///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_
#define CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_

#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ImpulseModel3DTpl : public ImpulseModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImpulseModelAbstractTpl<Scalar> Base;
  typedef ImpulseData3DTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImpulseDataAbstractTpl<Scalar> ImpulseDataAbstract;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs Matrix3s;
  typedef typename MathBase::MatrixXs MatrixXs;

  ImpulseModel3DTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                    const pinocchio::ReferenceFrame type = pinocchio::ReferenceFrame::LOCAL);
  virtual ~ImpulseModel3DTpl();

  virtual void calc(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  virtual void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  virtual void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const VectorXs& force);
  virtual boost::shared_ptr<ImpulseDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  DEPRECATED("Use get_id", pinocchio::FrameIndex get_frame() const { return id_; };)

  /**
   * @brief Print relevant information of the 3d impulse model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::id_;
  using Base::state_;
  using Base::type_;
};

template <typename _Scalar>
struct ImpulseData3DTpl : public ImpulseDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImpulseDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ImpulseData3DTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        dv0_local_dq(3, model->get_state()->get_nv()),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        v_partial_dv(6, model->get_state()->get_nv()) {
    frame = model->get_id();
    jMf = model->get_state()->get_pinocchio()->frames[model->get_id()].placement;
    fXj = jMf.inverse().toActionMatrix();
    v0_world.setZero();
    dv0_local_dq.setZero();
    fJf.setZero();
    v_partial_dq.setZero();
    v_partial_dv.setZero();
    v0_world_skew.setZero();
  }

  using Base::df_dx;
  using Base::dv0_dq;
  using Base::f;
  using Base::frame;
  using Base::Jc;
  using Base::jMf;
  using Base::pinocchio;

  typename pinocchio::SE3Tpl<Scalar>::ActionMatrixType fXj;
  Vector3s v0_world;
  Matrix3xs dv0_local_dq;
  Matrix6xs fJf;
  Matrix6xs v_partial_dq;
  Matrix6xs v_partial_dv;
  Matrix3s v0_world_skew;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulses/impulse-3d.hxx"

#endif  // CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_
