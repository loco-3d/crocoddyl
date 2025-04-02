///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, University of Edinburgh
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

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyInContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorActMultibodyInContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorJointActMultibodyInContactTpl)

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_
