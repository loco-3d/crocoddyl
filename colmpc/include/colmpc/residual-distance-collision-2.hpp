// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#ifndef COLMPC_RESIDUAL_DISTANCE_COLLISION_2_HPP_
#define COLMPC_RESIDUAL_DISTANCE_COLLISION_2_HPP_

#include "colmpc/fwd.hpp"
// include fwd first

#include <coal/collision_data.h>
#include <coal/distance.h>

#include <Eigen/Core>
#include <colmpc/data/geometry-data.hpp>
#include <colmpc/multibody.hpp>
#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/fcl.hpp>

#include "crocoddyl/core/utils/exception.hpp"

namespace colmpc {
using namespace crocoddyl;

template <typename _Scalar>
struct ResidualDistanceCollision2Tpl
    : public ResidualModelAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataDistanceCollision2Tpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::GeometryModel GeometryModel;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;
  /**
   * @brief Initialize the pair collision residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] nu          Dimension of the control vector
   * @param[in] pair_id     Index of the collision pair in the geometry model
   */

  ResidualDistanceCollision2Tpl(std::shared_ptr<StateMultibody> state,
                                const std::size_t nu,
                                const pinocchio::PairIndex pair_id);
  virtual ~ResidualDistanceCollision2Tpl();

  /**
   * @brief Compute the pair collision residual
   *
   * @param[in] data  Pair collision residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the pair collision residual
   *
   * @param[in] data  Pair collision residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Return the reference collision pair id
   */
  pinocchio::PairIndex get_pair_id() const;

  /**
   * @brief Modify the reference collision pair id
   */
  void set_pair_id(const pinocchio::PairIndex pair_id);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;
  using Base::v_dependent_;

 private:
  pinocchio::PairIndex
      pair_id_;  //!< Index of the collision pair in geometry model
  pinocchio::JointIndex joint_id_;  //!< Index of joint on which the collision
                                    //!< body frame of the robot is attached

  // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
  // q;
};

template <typename _Scalar>
struct ResidualDataDistanceCollision2Tpl
    : public ResidualDataAbstractTpl<_Scalar>,
      public GeometryDataWrapper {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;

  ResidualDataDistanceCollision2Tpl(
      ResidualDistanceCollision2Tpl<Scalar>* const model,
      DataCollectorAbstract* const data)
      : Base(model, data),
        J1(6, model->get_state()->get_nv()),
        J2(6, model->get_state()->get_nv()) {
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d =
        dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorActMultibodyTpl");
    }
    // Avoids data casting at runtime
    pinocchio = d->pinocchio;

    J1.fill(0);
    J2.fill(0);
    cp1.fill(0);
    cp2.fill(0);
    f1p1.fill(0);
    f2p2.fill(0);
  }
  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  Matrix6xs J1;
  Matrix6xs J2;
  Vector3s cp1;
  Vector3s cp2;

  Vector3s f1p1;
  pinocchio::SE3 f1Mp1;

  Vector3s f2p2;
  pinocchio::SE3 f2Mp2;

  pinocchio::SE3 oMg_id_1;
  pinocchio::SE3 oMg_id_2;
};

}  // namespace colmpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "colmpc/residual-distance-collision-2.hxx"
#endif  // COLMPC_RESIDUAL_DISTANCE_COLLISION_2_HPP_
