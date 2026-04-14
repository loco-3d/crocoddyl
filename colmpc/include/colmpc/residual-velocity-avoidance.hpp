// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#ifndef COLMPC_RESIDUAL_VELOCITY_AVOIDANCE_HPP_
#define COLMPC_RESIDUAL_VELOCITY_AVOIDANCE_HPP_

#include "colmpc/fwd.hpp"
// include fwd first

#include <coal/collision_data.h>
#include <coal/distance.h>

#include <Eigen/Core>
#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/fwd.hpp>
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
struct ResidualModelVelocityAvoidanceTpl
    : public ResidualModelAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataVelocityAvoidanceTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::GeometryModel GeometryModel;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix2s Matrix2s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef Eigen::DiagonalMatrix<Scalar, 3> DiagonalMatrix3s;
  typedef Eigen::Matrix<Scalar, 12, 1> Vector12s;
  typedef Eigen::Matrix<Scalar, 6, 6> Matrix6s;
  typedef Eigen::Matrix<Scalar, 8, 8> Matrix8s;
  typedef Eigen::Matrix<Scalar, 8, 6> Matrix86s;
  typedef Eigen::Matrix<Scalar, 8, 12> Matrix812s;
  typedef Eigen::Matrix<Scalar, 3, 6> Matrix36s;
  typedef Eigen::Matrix<Scalar, 6, 2> Matrix62s;
  typedef Eigen::Matrix<Scalar, 3, 12> Matrix312s;
  typedef Eigen::Matrix<Scalar, 12, 12> Matrix12s;
  typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic> Matrix6xLike;
  typedef Eigen::Matrix<Scalar, 12, Eigen::Dynamic> Matrix12xLike;
  /**
   * @brief Initialize the pair collision residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] nr          Dimension of residual vector
   * @param[in] geom_model  Pinocchio geometry model containing the collision
   * pair
   * @param[in] pair_id     Index of the collision pair in the geometry model
   * @param[in] di          Distance at which the robot starts to slow down
   * @param[in] ds          Security distance
   * @param[in] ksi         Convergence speed coefficient
   */

  ResidualModelVelocityAvoidanceTpl(
      std::shared_ptr<crocoddyl::StateMultibody> state,
      std::shared_ptr<GeometryModel> geom_model,
      const pinocchio::PairIndex pair_id, const Scalar di = 1.0e-2,
      const Scalar ds = 1.0e-5, const Scalar ksi = 1.0e-2);
  virtual ~ResidualModelVelocityAvoidanceTpl();

  /**
   * @brief Compute the pair collision residual
   *
   * @param[in] data  Pair collision residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the pair collision residual
   *
   * @param[in] data  Pair collision residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Return the Pinocchio geometry model
   */
  const GeometryModel& get_geometry() const;

  /**
   * @brief Return the reference collision pair id
   */
  pinocchio::PairIndex get_pair_id() const;

  /**
   * @brief Modify the reference collision pair id
   */
  void set_pair_id(const pinocchio::PairIndex pair_id);

  /**
   * @brief Return the distance at which the robot starts to slow down
   */
  Scalar get_di() const;

  /**
   * @brief Modify the distance at which the robot starts to slow down
   */
  void set_di(const Scalar di);

  /**
   * @brief Return the security distance
   */
  Scalar get_ds() const;

  /**
   * @brief Modify the security distance
   */
  void set_ds(const Scalar ds);

  /**
   * @brief Return the convergence speed coefficient
   */
  Scalar get_ksi() const;

  /**
   * @brief Modify the convergence speed coefficient
   */
  void set_ksi(const Scalar ksi);

 protected:
  using Base::nr_;
  using Base::state_;
  using Base::unone_;
  using Base::v_dependent_;

 private:
  /**
   * @brief Evaluate type of geometry object and create corresponding D matrix
   * @param[in] geom Collision geometry object to obtain D matrix
   */
  inline DiagonalMatrix3s cast_geom_to_d(
      const std::shared_ptr<coal::CollisionGeometry>& geom);

  typename StateMultibody::PinocchioModel
      pin_model_;  //!< Pinocchio model used for internal computations
  std::shared_ptr<GeometryModel>
      geom_model_;  //!< Pinocchio geometry model containing collision pair
  pinocchio::PairIndex
      pair_id_;  //!< Index of the collision pair in geometry model

  DiagonalMatrix3s D1_inv_pow_;
  DiagonalMatrix3s D2_inv_pow_;

  Scalar di_;   //!< Distance at which the robot starts to slow down
  Scalar ds_;   //!< Security distance
  Scalar ksi_;  //!< Convergence speed coefficient
};

template <typename _Scalar>
struct ResidualDataVelocityAvoidanceTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  typedef pinocchio::DataTpl<Scalar> PinocchioData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef Eigen::Matrix<Scalar, 8, 8> Matrix8s;
  typedef Eigen::Matrix<Scalar, 8, 6> Matrix86s;
  typedef Eigen::Matrix<Scalar, 3, 6> Matrix36s;
  typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic> Matrix6xLike;
  typedef Eigen::Matrix<Scalar, 12, Eigen::Dynamic> Matrix12xLike;

  template <template <typename Scalar> class Model>
  ResidualDataVelocityAvoidanceTpl(Model<Scalar>* const model,
                                   DataCollectorAbstract* const data)
      : Base(model, data),
        geometry(pinocchio::GeometryData(model->get_geometry())),
        req(),
        res(),
        ddistdot_dq_val(model->get_state()->get_nx()),
        d_dist_dot_dq(model->get_state()->get_nq()),
        J(model->get_state()->get_nq()),
        q(model->get_state()->get_nq()),
        v(model->get_state()->get_nq()),
        __a(model->get_state()->get_nq()),
        J1(6, model->get_state()->get_nq()),
        J2(6, model->get_state()->get_nq()),
        jacobian1(6, model->get_state()->get_nq()),
        jacobian2(6, model->get_state()->get_nq()),
        in1_dnu1_dq(6, model->get_state()->get_nq()),
        in2_dnu2_dq(6, model->get_state()->get_nq()),
        in1_dnu1_dqdot(6, model->get_state()->get_nq()),
        in2_dnu2_dqdot(6, model->get_state()->get_nq()),
        d_theta_dq(12, model->get_state()->get_nq()),
        d_theta_dot_dq(12, model->get_state()->get_nq()) {
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

    ddistdot_dq_val.setZero();
    d_dist_dot_dq.setZero();
    J.setZero();
    q.setZero();
    v.setZero();
    __a.setZero();
    J1.setZero();
    J2.setZero();
    jacobian1.setZero();
    jacobian2.setZero();
    in1_dnu1_dq.setZero();
    in2_dnu2_dq.setZero();
    in1_dnu1_dqdot.setZero();
    in2_dnu2_dqdot.setZero();
    d_theta_dq.setZero();
    d_theta_dot_dq.setZero();

    oMg_id_1.setIdentity();
    oMg_id_2.setIdentity();
    f1Mp1.setIdentity();
    f2Mp2.setIdentity();

    Lyc.setZero();
    Lyr.setZero();
  }
  pinocchio::GeometryData geometry;       //!< Pinocchio geometry data
  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data

  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  coal::DistanceRequest req;  //!< Distance Request from coal,
                              //!< used to compute the distance between shapes
  coal::DistanceResult res;   //!< Distance Result from coal

  pinocchio::SE3 oMg_id_1;
  pinocchio::SE3 oMg_id_2;

  pinocchio::SE3 f1Mp1;
  pinocchio::SE3 f2Mp2;

  Scalar Ldot;
  Scalar distance;
  Scalar distance_inv;

  Vector3s x_diff;
  Vector3s x1_c1_diff;
  Vector3s x2_c2_diff;
  Vector3s x_diff_cross_x1_c1_diff;
  Vector3s x_diff_cross_x2_c2_diff;

  pinocchio::Motion m1;
  pinocchio::Motion m2;

  VectorXs ddistdot_dq_val;
  VectorXs d_dist_dot_dq;
  VectorXs J;
  VectorXs q;
  VectorXs v;
  VectorXs __a;
  Matrix6xLike J1;
  Matrix6xLike J2;
  Matrix6xLike jacobian1;
  Matrix6xLike jacobian2;
  Matrix6xLike in1_dnu1_dq;
  Matrix6xLike in2_dnu2_dq;
  Matrix6xLike in1_dnu1_dqdot;
  Matrix6xLike in2_dnu2_dqdot;
  Matrix12xLike d_theta_dq;
  Matrix12xLike d_theta_dot_dq;

  Matrix86s Lyc;
  Matrix86s Lyr;
};

}  // namespace colmpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "colmpc/residual-velocity-avoidance.hxx"
#endif  // COLMPC_RESIDUAL_VELOCITY_AVOIDANCE_HPP_
