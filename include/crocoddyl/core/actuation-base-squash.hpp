///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh,
//                          Universitat Politecnica de Catalunya
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATION_BASE_SQUASH_HPP_
#define CROCODDYL_CORE_ACTIVATION_BASE_SQUASH_HPP_

#include <Eigen/Dense>
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

class ActuationModelSquashingAbstract : public ActuationModelAbstract {
 public:
  ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state, const std::size_t& nu);
  virtual ~ActuationModelSquashingAbstract();

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                        const bool& recalc = true) = 0;

  virtual void calcSquash(const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcSquashDiff(const Eigen::Ref<const Eigen::VectorXd>& u) = 0;

 protected:
  Eigen::VectorXd v_;      // Squashing function output
  Eigen::MatrixXd dv_du_;  // Squashing function Jacobian wrt control input u
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_BASE_SQUASH_HPP_
