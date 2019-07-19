///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_STATE_MULTIBODY_HPP_
#define CROCODDYL_MULTIBODY_STATES_STATE_MULTIBODY_HPP_

#include "crocoddyl/core/state-base.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class StateMultibody : public StateAbstract {
 public:
  StateMultibody(pinocchio::Model& model);
  ~StateMultibody();

  Eigen::VectorXd zero() override;
  Eigen::VectorXd rand() override;
  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout) override;
  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> xout) override;
  void Jdiff(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
             Jcomponent firstsecond = both) override;
  void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                  Jcomponent firstsecond = both) override;

 private:
  pinocchio::Model model_;
  Eigen::VectorXd x0_;
  Eigen::VectorXd dx_;
  Eigen::MatrixXd Jdq_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_STATES_STATE_MULTIBODY_HPP_
