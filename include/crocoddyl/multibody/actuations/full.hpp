///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_

#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

class ActuationModelFull : public ActuationModelAbstract {
 public:
  explicit ActuationModelFull(StateMultibody& state);
  ~ActuationModelFull();

  void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true) = 0;
  boost::shared_ptr<ActuationDataAbstract> createData();
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
