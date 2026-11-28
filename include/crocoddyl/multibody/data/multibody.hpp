///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_MULTIBODY_HPP_
#define CROCODDYL_CORE_DATA_MULTIBODY_HPP_

#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/data/actuation.hpp"
#include "crocoddyl/core/data/joint.hpp"
#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorMultibodyTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyTpl(pinocchio::DataTpl<Scalar>* const data)
      : pinocchio(data) {}
  virtual ~DataCollectorMultibodyTpl() {}

  pinocchio::DataTpl<Scalar>* pinocchio;
};

template <typename Scalar>
struct DataCollectorMultibodyParamsTpl : DataCollectorMultibodyTpl<Scalar>,
                                         DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorMultibodyParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyTpl : DataCollectorMultibodyTpl<Scalar>,
                                      DataCollectorActuationTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActMultibodyTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio),
        DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyParamsTpl
    : DataCollectorActMultibodyTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActMultibodyParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorActMultibodyTpl<Scalar>(pinocchio, actuation),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorActMultibodyParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActMultibodyTpl : DataCollectorActMultibodyTpl<Scalar>,
                                           DataCollectorJointTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActMultibodyTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint)
      : DataCollectorActMultibodyTpl<Scalar>(pinocchio, actuation),
        DataCollectorJointTpl<Scalar>(joint) {}
  virtual ~DataCollectorJointActMultibodyTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActMultibodyParamsTpl
    : DataCollectorJointActMultibodyTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActMultibodyParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorJointActMultibodyTpl<Scalar>(pinocchio, actuation, joint),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorJointActMultibodyParamsTpl() {}
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorMultibodyTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyParamsTpl)

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_HPP_
