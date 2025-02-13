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

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"

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

  /**
   * @brief Initialize the 3d impulse model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id of the impulse
   * @param[in] type   Type of impulse (default LOCAL)
   */
  ImpulseModel3DTpl(
      boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
      const pinocchio::ReferenceFrame type = pinocchio::ReferenceFrame::LOCAL);
  virtual ~ImpulseModel3DTpl();

  /**
   * @brief Compute the 6d impulse Jacobian
   *
   * @param[in] data  6d impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the 6d impulse holonomic constraint
   *
   * @param[in] data  6d impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Convert the force into a stack of spatial forces
   *
   * @param[in] data   6d impulse data
   * @param[in] force  6d impulse
   */
  virtual void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data,
                           const VectorXs& force);

  /**
   * @brief Create the 6d impulse data
   */
  virtual boost::shared_ptr<ImpulseDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data);

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
  typedef typename pinocchio::ForceTpl<Scalar> Force;

  template <template <typename Scalar> class Model>
  ImpulseData3DTpl(Model<Scalar>* const model,
                   pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        f_local(Force::Zero()),
        dv0_local_dq(3, model->get_state()->get_nv()),
        fJf_df(3, model->get_state()->get_nv()) {
    // There is only one element in the force_datas vector
    ForceDataAbstract& fdata = force_datas[0];
    fdata.frame = model->get_id();
    fdata.jMf = model->get_state()->get_pinocchio()->frames[fdata.frame].placement;
    fdata.fXj = fdata.jMf.inverse().toActionMatrix();

    v0.setZero();
    dv0_local_dq.setZero();
    v0_skew.setZero();
    v0_world_skew.setZero();
    f_skew.setZero();
    fJf_df.setZero();
  }

  using Base::df_dx;
  using Base::dv0_dq;
  using Base::Jc;
  using Base::pinocchio;
  using Base::force_datas;

  Vector3s v0;
  Force f_local;
  Matrix3xs dv0_local_dq;
  Matrix3s v0_skew;
  Matrix3s v0_world_skew;
  Matrix3s f_skew;
  Matrix3xs fJf_df;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulses/impulse-3d.hxx"

#endif  // CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_
