///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_IMPLICIT_CONSTRAINTS_HPP_
#define CROCODDYL_CORE_DATA_IMPLICIT_CONSTRAINTS_HPP_

#include "crocoddyl/core/data/observer.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorImplicitConstraintTpl
    : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit DataCollectorImplicitConstraintTpl(
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints)
      : DataCollectorAbstractTpl<Scalar>(), constraints(constraints) {}
  virtual ~DataCollectorImplicitConstraintTpl() {}

  std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints;
};

template <typename Scalar>
struct DataCollectorMultibodyInImplicitConstraintTpl
    : DataCollectorMultibodyTpl<Scalar>,
      DataCollectorImplicitConstraintTpl<Scalar>,
      DataCollectorObserverTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyInImplicitConstraintTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio),
        DataCollectorImplicitConstraintTpl<Scalar>(constraints),
        DataCollectorObserverTpl<Scalar>() {}
  virtual ~DataCollectorMultibodyInImplicitConstraintTpl() {}
};

template <typename Scalar>
struct DataCollectorMultibodyInImplicitConstraintParamsTpl
    : DataCollectorMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyInImplicitConstraintParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorMultibodyInImplicitConstraintTpl<Scalar>(pinocchio,
                                                              constraints),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorMultibodyInImplicitConstraintParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyInImplicitConstraintTpl
    : DataCollectorMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorActuationTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActMultibodyInImplicitConstraintTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints)
      : DataCollectorMultibodyInImplicitConstraintTpl<Scalar>(pinocchio,
                                                              constraints),
        DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyInImplicitConstraintTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyInImplicitConstraintParamsTpl
    : DataCollectorActMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActMultibodyInImplicitConstraintParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorActMultibodyInImplicitConstraintTpl<Scalar>(
            pinocchio, actuation, constraints),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorActMultibodyInImplicitConstraintParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorJointMultibodyInImplicitConstraintTpl
    : DataCollectorMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorJointTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointMultibodyInImplicitConstraintTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints)
      : DataCollectorMultibodyInImplicitConstraintTpl<Scalar>(pinocchio,
                                                              constraints),
        DataCollectorJointTpl<Scalar>(joint) {}
  virtual ~DataCollectorJointMultibodyInImplicitConstraintTpl() {}
};

template <typename Scalar>
struct DataCollectorJointMultibodyInImplicitConstraintParamsTpl
    : DataCollectorJointMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointMultibodyInImplicitConstraintParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorJointMultibodyInImplicitConstraintTpl<Scalar>(
            pinocchio, joint, constraints),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorJointMultibodyInImplicitConstraintParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActMultibodyInImplicitConstraintTpl
    : DataCollectorActMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorJointTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActMultibodyInImplicitConstraintTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints)
      : DataCollectorActMultibodyInImplicitConstraintTpl<Scalar>(
            pinocchio, actuation, constraints),
        DataCollectorJointTpl<Scalar>(joint) {}
  virtual ~DataCollectorJointActMultibodyInImplicitConstraintTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActMultibodyInImplicitConstraintParamsTpl
    : DataCollectorJointActMultibodyInImplicitConstraintTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActMultibodyInImplicitConstraintParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ImplicitConstraintDataMultipleTpl<Scalar> > constraints,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params,
      ParameterDataManagerTpl<Scalar>* const parameter_data = nullptr)
      : DataCollectorJointActMultibodyInImplicitConstraintTpl<Scalar>(
            pinocchio, actuation, joint, constraints),
        DataCollectorParamsTpl<Scalar>(params, parameter_data) {}
  virtual ~DataCollectorJointActMultibodyInImplicitConstraintParamsTpl() {}
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorImplicitConstraintTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyInImplicitConstraintTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyInImplicitConstraintParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyInImplicitConstraintTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyInImplicitConstraintParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointMultibodyInImplicitConstraintTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointMultibodyInImplicitConstraintParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyInImplicitConstraintTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyInImplicitConstraintParamsTpl)

#endif  // CROCODDYL_CORE_DATA_IMPLICIT_CONSTRAINTS_HPP_
