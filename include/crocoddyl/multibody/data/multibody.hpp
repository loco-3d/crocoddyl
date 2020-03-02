///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_MULTIBODY_HPP_
#define CROCODDYL_CORE_DATA_MULTIBODY_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/data/actuation.hpp"

#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorMultibodyTpl : virtual DataCollectorAbstractTpl<Scalar> {
  DataCollectorMultibodyTpl(pinocchio::DataTpl<Scalar>* const data) : pinocchio(data) {}
  virtual ~DataCollectorMultibodyTpl() {}

  pinocchio::DataTpl<Scalar>* pinocchio;
};

template <typename Scalar>
struct DataCollectorActMultibodyTpl : DataCollectorMultibodyTpl<Scalar>, DataCollectorActuationTpl<Scalar> {
  DataCollectorActMultibodyTpl(pinocchio::DataTpl<Scalar>* const pinocchio,
                               boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation)
      : DataCollectorMultibodyTpl<Scalar>(pinocchio), DataCollectorActuationTpl<Scalar>(actuation) {}
  virtual ~DataCollectorActMultibodyTpl() {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_HPP_
