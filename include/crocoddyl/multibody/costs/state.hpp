///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_STATE_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

class CostModelState : public CostModelAbstract {
 public:
  CostModelState(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::VectorXd& xref,
                 const unsigned int& nu);
  CostModelState(StateMultibody& state, ActivationModelAbstract& activation, const Eigen::VectorXd& xref);
  CostModelState(StateMultibody& state, const Eigen::VectorXd& xref, const unsigned int& nu);
  CostModelState(StateMultibody& state, const Eigen::VectorXd& xref);
  CostModelState(StateMultibody& state, ActivationModelAbstract& activation, const unsigned int& nu);
  CostModelState(StateMultibody& state, const unsigned int& nu);
  CostModelState(StateMultibody& state, ActivationModelAbstract& activation);
  CostModelState(StateMultibody& state);

  ~CostModelState();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

  const Eigen::VectorXd& get_xref() const;

 private:
  Eigen::VectorXd xref_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
