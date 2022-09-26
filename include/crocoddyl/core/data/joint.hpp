///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_JOINT_HPP_
#define CROCODDYL_CORE_DATA_JOINT_HPP_

#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/data/actuation.hpp"
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
   * @brief Initialize a joint data structure
   *
   * @param state      State description
   * @param actuation  Actuation model
   * @param nu         Dimension of control input
   */
  JointDataAbstractTpl(boost::shared_ptr<StateAbstract> state, boost::shared_ptr<ActuationModelAbstract> actuation,
                       const std::size_t nu)
      : tau(actuation->get_nu()),
        a(state->get_nv()),
        dtau_dx(actuation->get_nu(), state->get_ndx()),
        dtau_du(actuation->get_nu(), nu),
        da_dx(state->get_nv(), state->get_ndx()),
        da_du(state->get_nv(), nu) {
    tau.setZero();
    a.setZero();
    dtau_dx.setZero();
    dtau_du.setZero();
    da_dx.setZero();
    da_du.setZero();
  }
  virtual ~JointDataAbstractTpl() {}

  VectorXs tau;      //!< Joint torque commands
  VectorXs a;        //!< Generalized joint acceleration
  MatrixXs dtau_dx;  //!< Partial derivatives of the joint torques w.r.t. the state point
  MatrixXs dtau_du;  //!< Partial derivatives of the joint torques w.r.t. the control input
  MatrixXs da_dx;    //!< Partial derivatives of the generalized joint accelerations w.r.t. the state point
  MatrixXs da_du;    //!< Partial derivatives of the generalized joint accelerations w.r.t. the control input
};

template <typename Scalar>
struct DataCollectorJointTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointTpl(boost::shared_ptr<JointDataAbstractTpl<Scalar> > joint)
      : DataCollectorAbstractTpl<Scalar>(), joint(joint) {}
  virtual ~DataCollectorJointTpl() {}

  boost::shared_ptr<JointDataAbstractTpl<Scalar> > joint;
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
  DataCollectorJointActuationTpl(boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
                                 boost::shared_ptr<JointDataAbstractTpl<Scalar> > joint)
      : DataCollectorActuationTpl<Scalar>(actuation), joint(joint) {}
  virtual ~DataCollectorJointActuationTpl() {}

  boost::shared_ptr<JointDataAbstractTpl<Scalar> > joint;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_JOINT_HPP_
