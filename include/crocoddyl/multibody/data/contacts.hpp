///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_CONTACTS_HPP_
#define CROCODDYL_CORE_DATA_CONTACTS_HPP_

#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorContactTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorContactTpl(
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorAbstractTpl<Scalar>(), contacts(contacts) {}
  virtual ~DataCollectorContactTpl() {}

  std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts;
};

template <typename Scalar>
struct DataCollectorMultibodyInContactTpl : DataCollectorMultibodyTpl<Scalar>,
                                            DataCollectorContactTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyInContactTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio),
        DataCollectorContactTpl<Scalar>(contacts) {}
  virtual ~DataCollectorMultibodyInContactTpl() {}
};

template <typename Scalar>
struct DataCollectorMultibodyInContactParamsTpl
    : DataCollectorMultibodyInContactTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyInContactParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params)
      : DataCollectorMultibodyInContactTpl<Scalar>(pinocchio, contacts),
        DataCollectorParamsTpl<Scalar>(params) {}
  virtual ~DataCollectorMultibodyInContactParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyInContactTpl
    : DataCollectorMultibodyInContactTpl<Scalar>,
      DataCollectorActuationTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActMultibodyInContactTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorMultibodyInContactTpl<Scalar>(pinocchio, contacts),
        DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyInContactTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyInContactParamsTpl
    : DataCollectorActMultibodyInContactTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorActMultibodyInContactParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params)
      : DataCollectorActMultibodyInContactTpl<Scalar>(pinocchio, actuation,
                                                      contacts),
        DataCollectorParamsTpl<Scalar>(params) {}
  virtual ~DataCollectorActMultibodyInContactParamsTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActMultibodyInContactTpl
    : DataCollectorActMultibodyInContactTpl<Scalar>,
      DataCollectorJointTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActMultibodyInContactTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts)
      : DataCollectorActMultibodyInContactTpl<Scalar>(pinocchio, actuation,
                                                      contacts),
        DataCollectorJointTpl<Scalar>(joint) {}
  virtual ~DataCollectorJointActMultibodyInContactTpl() {}
};

template <typename Scalar>
struct DataCollectorJointActMultibodyInContactParamsTpl
    : DataCollectorJointActMultibodyInContactTpl<Scalar>,
      DataCollectorParamsTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorJointActMultibodyInContactParamsTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
      std::shared_ptr<JointDataAbstractTpl<Scalar> > joint,
      std::shared_ptr<ContactDataMultipleTpl<Scalar> > contacts,
      std::shared_ptr<ParamsDataAbstractTpl<Scalar> > params)
      : DataCollectorJointActMultibodyInContactTpl<Scalar>(pinocchio, actuation,
                                                           joint, contacts),
        DataCollectorParamsTpl<Scalar>(params) {}
  virtual ~DataCollectorJointActMultibodyInContactParamsTpl() {}
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyInContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyInContactParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyInContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyInContactParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyInContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyInContactParamsTpl)

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_
