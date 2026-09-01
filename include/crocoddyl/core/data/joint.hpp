///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_JOINT_HPP_
#define CROCODDYL_CORE_DATA_JOINT_HPP_

#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/data/actuation.hpp"
#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct JointDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize joint data from its effort and control dimensions
   *
   * @param state  State description
   * @param ntau   Dimension of joint efforts
   * @param nu     Dimension of control input
   */
  JointDataAbstractTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t ntau, const std::size_t nu)
      : tau(ntau),
        a(state->get_nv()),
        dtau_dx(ntau, state->get_ndx()),
        dtau_du(ntau, nu),
        da_dx(state->get_nv(), state->get_ndx()),
        da_du(state->get_nv(), nu) {
    tau.setZero();
    a.setZero();
    dtau_dx.setZero();
    dtau_du.setZero();
    da_dx.setZero();
    da_du.setZero();
  }

  /**
   * @brief Initialize a joint data structure containing generalized
   * accelerations and joint efforts, and their derivatives.
   *
   * @param state      State description
   * @param actuation  Actuation model
   * @param nu         Dimension of control input
   */
  JointDataAbstractTpl(std::shared_ptr<StateAbstract> state,
                       std::shared_ptr<ActuationModelAbstract> actuation,
                       const std::size_t nu)
      : JointDataAbstractTpl(state, actuation->get_nu(), nu) {}
  virtual ~JointDataAbstractTpl() {}

  VectorXs tau;      //!< Joint efforts
  VectorXs a;        //!< Generalized joint acceleration
  MatrixXs dtau_dx;  //!< Partial derivatives of the joint efforts w.r.t. the
                     //!< state point
  MatrixXs dtau_du;  //!< Partial derivatives of the joint efforts w.r.t. the
                     //!< control input
  MatrixXs da_dx;    //!< Partial derivatives of the generalized joint
                     //!< accelerations w.r.t. the state point
  MatrixXs da_du;    //!< Partial derivatives of the generalized joint
                     //!< accelerations w.r.t. the control input
};

template <typename Scalar>
struct DataCollectorJointTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointTpl(std::shared_ptr<JointDataAbstractTpl<Scalar> > joint)
      : DataCollectorAbstractTpl<Scalar>(), joint(joint) {}
  virtual ~DataCollectorJointTpl() {}

  std::shared_ptr<JointDataAbstractTpl<Scalar> > joint;
};

template <typename Scalar>
struct DataCollectorJointParamsTpl : DataCollectorJointTpl<Scalar>,
                                     DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointParamsTpl(
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorJointTpl<Scalar>(joint),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorJointParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActuationTpl : DataCollectorActuationTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the joint-actuation data collector
   *
   * @param[in] actuation  Actuation data
   * @param[in] joint      Joint data
   */
  DataCollectorJointActuationTpl(
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint)
      : DataCollectorActuationTpl<Scalar>(actuation), joint(joint) {}
  virtual ~DataCollectorJointActuationTpl() {}

  std::shared_ptr<JointDataAbstractTpl<Scalar> > joint;
};

template <typename Scalar>
struct DataCollectorJointActuationParamsTpl
    : DataCollectorJointActuationTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActuationParamsTpl(
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorJointActuationTpl<Scalar>(actuation, joint),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorJointActuationParamsTpl() {}
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::JointDataAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorJointTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorJointParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActuationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActuationParamsTpl)

#endif  // CROCODDYL_CORE_DATA_JOINT_HPP_
