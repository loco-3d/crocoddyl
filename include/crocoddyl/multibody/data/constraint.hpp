///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_CONSTRAINTS_HPP_
#define CROCODDYL_CORE_DATA_CONSTRAINTS_HPP_

#if PINOCCHIO_VERSION_AT_LEAST(2,9,0)

#include <boost/shared_ptr.hpp>

#include <pinocchio/algorithm/contact-info.hpp>
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorConstraintTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef pinocchio::RigidConstraintDataTpl<Scalar, 0> RigidConstraintData;

  DataCollectorConstraintTpl<Scalar>(const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData) & contacts)
      : DataCollectorAbstractTpl<Scalar>(), contacts(contacts) {}
  virtual ~DataCollectorConstraintTpl() {}
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData) contacts;
};

template <typename Scalar>
struct DataCollectorMultibodyInConstraintTpl : DataCollectorMultibodyTpl<Scalar>, DataCollectorConstraintTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef pinocchio::RigidConstraintDataTpl<Scalar, 0> RigidConstraintData;
  DataCollectorMultibodyInConstraintTpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                                        PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData) contacts)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio), DataCollectorConstraintTpl<Scalar>(contacts) {}
  virtual ~DataCollectorMultibodyInConstraintTpl() {}
};

template <typename Scalar>
struct DataCollectorActMultibodyInConstraintTpl : DataCollectorMultibodyInConstraintTpl<Scalar>,
                                                  DataCollectorActuationTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef pinocchio::RigidConstraintDataTpl<Scalar, 0> RigidConstraintData;
  DataCollectorActMultibodyInConstraintTpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                                           boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation,
                                           PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData) contacts)
      : DataCollectorMultibodyInConstraintTpl<Scalar>(pinocchio, contacts),
        DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyInConstraintTpl() {}
};

}  // namespace crocoddyl

#endif //PINOCCHIO_VERSION_AT_LEAST(2,9,0)

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_CONSTRAINTS_HPP_
