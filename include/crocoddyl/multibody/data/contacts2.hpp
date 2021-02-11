///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_CONTACTS2_HPP_
#define CROCODDYL_CORE_DATA_CONTACTS2_HPP_

#include <boost/shared_ptr.hpp>

#include <pinocchio/algorithm/contact-info.hpp>
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorContact2Tpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef pinocchio::RigidContactDataTpl<Scalar,0> RigidContactData;
  
  DataCollectorContact2Tpl<Scalar>(const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData)& contacts)
      : DataCollectorAbstractTpl<Scalar>(), contacts(contacts) {}
  virtual ~DataCollectorContact2Tpl() {}
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contacts;
};

template <typename Scalar>
struct DataCollectorMultibodyInContact2Tpl : DataCollectorMultibodyTpl<Scalar>, DataCollectorContact2Tpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef pinocchio::RigidContactDataTpl<Scalar,0> RigidContactData;
  DataCollectorMultibodyInContact2Tpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                                     PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contacts)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio),
    DataCollectorContact2Tpl<Scalar>(contacts) {}
  virtual ~DataCollectorMultibodyInContact2Tpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyInContact2Tpl : DataCollectorMultibodyInContact2Tpl<Scalar>,
                                               DataCollectorActuationTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef pinocchio::RigidContactDataTpl<Scalar,0> RigidContactData;
  DataCollectorActMultibodyInContact2Tpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                                        boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
                                        PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contacts)
      : DataCollectorMultibodyInContact2Tpl<Scalar>(pinocchio, contacts),
        DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyInContact2Tpl() {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONTACT_HPP_
