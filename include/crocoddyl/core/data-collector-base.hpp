///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DATA_COLLECTOR_BASE_HPP_
#define CROCODDYL_CORE_DATA_COLLECTOR_BASE_HPP_

#include <Eigen/Dense>
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename Scalar>
struct DataCollectorAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataCollectorAbstractTpl() {}
  virtual ~DataCollectorAbstractTpl() {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_COLLECTOR_BASE_HPP_
