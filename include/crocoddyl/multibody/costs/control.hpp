///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

class CostModelControl : public CostModelAbstract {
 public:
  CostModelControl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                   const Eigen::VectorXd& uref);
  CostModelControl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation);
  CostModelControl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                   const std::size_t& nu);
  CostModelControl(boost::shared_ptr<StateMultibody> state, const Eigen::VectorXd& uref);
  explicit CostModelControl(boost::shared_ptr<StateMultibody> state);
  CostModelControl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  ~CostModelControl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

  const Eigen::VectorXd& get_uref() const;
  void set_uref(const Eigen::VectorXd& uref_in);

 private:
  Eigen::VectorXd uref_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
