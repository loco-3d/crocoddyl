///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh,
//                          Universitat Politecnica de Catalunya
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_SQUASH_BASE_HPP_
#define CROCODDYL_CORE_ACTUATION_SQUASH_BASE_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/squashing-base.hpp"

namespace crocoddyl {

class ActuationModelSquashingAbstract : public ActuationModelAbstract {
 public:
  ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state, boost::shared_ptr<SquashingModelAbstract> squashing, const std::size_t& nu);
  virtual ~ActuationModelSquashingAbstract();

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                        const bool& recalc = true) = 0;
  
  const boost::shared_ptr<SquashingModelAbstract>& get_squashing() const;

 protected:
  boost::shared_ptr<SquashingModelAbstract> squashing_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_SQUASH_BASE_HPP_
