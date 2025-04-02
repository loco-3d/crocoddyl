///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_IMPULSES_HPP_
#define CROCODDYL_CORE_DATA_IMPULSES_HPP_

#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorImpulseTpl : virtual DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorImpulseTpl(
      std::shared_ptr<ImpulseDataMultipleTpl<Scalar> > impulses)
      : DataCollectorAbstractTpl<Scalar>(), impulses(impulses) {}
  virtual ~DataCollectorImpulseTpl() {}

  std::shared_ptr<ImpulseDataMultipleTpl<Scalar> > impulses;
};

template <typename Scalar>
struct DataCollectorMultibodyInImpulseTpl : DataCollectorMultibodyTpl<Scalar>,
                                            DataCollectorImpulseTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorMultibodyInImpulseTpl(
      pinocchio::DataTpl<Scalar>* const pinocchio,
      std::shared_ptr<ImpulseDataMultipleTpl<Scalar> > impulses)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio),
        DataCollectorImpulseTpl<Scalar>(impulses) {}
  virtual ~DataCollectorMultibodyInImpulseTpl() {}
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DataCollectorImpulseTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DataCollectorMultibodyInImpulseTpl)

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_IN_IMPULSE_HPP_
