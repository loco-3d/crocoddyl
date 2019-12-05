///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_MULTIBODY_HPP_
#define CROCODDYL_CORE_DATA_MULTIBODY_HPP_

#include <pinocchio/multibody/data.hpp>
#include "crocoddyl/core/data-collector-base.hpp"

namespace crocoddyl {

struct DataCollectorMultibody : virtual DataCollectorAbstract {
  DataCollectorMultibody(pinocchio::Data* const data) : pinocchio(data) {}
  virtual ~DataCollectorMultibody() {}

  pinocchio::Data* pinocchio;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_MULTIBODY_HPP_
