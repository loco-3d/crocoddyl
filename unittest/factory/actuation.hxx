///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "actuation.hpp"
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {
namespace unittest {

template <typename Scalar>
void updateActuation(
    const std::shared_ptr<crocoddyl::ActuationModelAbstractTpl<Scalar>>& model,
    const std::shared_ptr<crocoddyl::ActuationDataAbstractTpl<Scalar>>& data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& u) {
  model->calc(data, x, u);
}

}  // namespace unittest
}  // namespace crocoddyl
