///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ARM_FACTORY_HPP_
#define CROCODDYL_ARM_FACTORY_HPP_

#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace benchmark {

template <typename Scalar>
void build_arm_action_models(
    std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& runningModel,
    std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >&
        terminalModel) {
  typedef typename crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>
      DifferentialActionModelFreeFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar>
      IntegratedActionModelEuler;
  typedef typename crocoddyl::ActuationModelFullTpl<Scalar> ActuationModelFull;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef typename crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;
  typedef typename crocoddyl::ResidualModelFramePlacementTpl<Scalar>
      ResidualModelFramePlacement;
  typedef typename crocoddyl::ResidualModelControlTpl<Scalar>
      ResidualModelControl;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;

  // because urdf is not supported with all scalar types.
  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR
                              "/kinova_description/robots/kinova.urdf",
                              modeld);
  pinocchio::srdf::loadReferenceConfigurations(
      modeld,
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/kinova_description/srdf/kinova.srdf",
      false);

  pinocchio::ModelTpl<Scalar> model(modeld.cast<Scalar>());

  std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      std::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
          std::make_shared<pinocchio::ModelTpl<Scalar> >(model));

  std::shared_ptr<CostModelAbstract> goalTrackingCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelFramePlacement>(
                     state, model.getFrameId("j2s6s200_end_effector"),
                     pinocchio::SE3Tpl<Scalar>(
                         Matrix3s::Identity(),
                         Vector3s(Scalar(0.6), Scalar(0.2), Scalar(0.5)))));
  std::shared_ptr<CostModelAbstract> xRegCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelState>(state));
  std::shared_ptr<CostModelAbstract> uRegCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelControl>(state));

  // Create a cost model per the running and terminal action model.
  std::shared_ptr<CostModelSum> runningCostModel =
      std::make_shared<CostModelSum>(state);
  std::shared_ptr<CostModelSum> terminalCostModel =
      std::make_shared<CostModelSum>(state);

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-1));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-1));
  terminalCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1e3));

  // We define an actuation model
  std::shared_ptr<ActuationModelFull> actuation =
      std::make_shared<ActuationModelFull>(state);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  std::shared_ptr<DifferentialActionModelFreeFwdDynamics> runningDAM =
      std::make_shared<DifferentialActionModelFreeFwdDynamics>(
          state, actuation, runningCostModel);

  runningModel =
      std::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-2));
  terminalModel =
      std::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(0.));
}

}  // namespace benchmark
}  // namespace crocoddyl

#endif  // CROCODDYL_ARM_FACTORY_HPP_
