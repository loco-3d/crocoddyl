///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ARM_KINOVA_FACTORY_HPP_
#define CROCODDYL_ARM_KINOVA_FACTORY_HPP_

#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/costs/control.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {
namespace benchmark {

template <typename Scalar>
void build_arm_kinova_action_models(
    boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> &runningModel,
    boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>>
        &terminalModel) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::FramePlacementTpl<Scalar> FramePlacement;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelFramePlacementTpl<Scalar>
      CostModelFramePlacement;
  typedef typename crocoddyl::CostModelStateTpl<Scalar> CostModelState;
  typedef typename crocoddyl::CostModelControlTpl<Scalar> CostModelControl;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::ActuationModelFullTpl<Scalar> ActuationModelFull;
  typedef typename crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>
      DifferentialActionModelFreeFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar>
      IntegratedActionModelEuler;

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

  boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<Scalar>>(
          boost::make_shared<pinocchio::ModelTpl<Scalar>>(model));

  FramePlacement Mref(
      model.getFrameId("j2s6s200_end_effector"),
      pinocchio::SE3Tpl<Scalar>(Matrix3s::Identity(),
                                Vector3s(Scalar(0), Scalar(0), Scalar(.4))));
  boost::shared_ptr<CostModelAbstract> goalTrackingCost =
      boost::make_shared<CostModelFramePlacement>(state, Mref);
  boost::shared_ptr<CostModelAbstract> xRegCost =
      boost::make_shared<CostModelState>(state);
  boost::shared_ptr<CostModelAbstract> uRegCost =
      boost::make_shared<CostModelControl>(state);

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<CostModelSum> runningCostModel =
      boost::make_shared<CostModelSum>(state);
  boost::shared_ptr<CostModelSum> terminalCostModel =
      boost::make_shared<CostModelSum>(state);

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));
  terminalCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));

  // We define an actuation model
  boost::shared_ptr<ActuationModelFull> actuation =
      boost::make_shared<ActuationModelFull>(state);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<DifferentialActionModelFreeFwdDynamics> runningDAM =
      boost::make_shared<DifferentialActionModelFreeFwdDynamics>(
          state, actuation, runningCostModel);

  // VectorXs armature(state->get_nq());
  // armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  // runningDAM->set_armature(armature);
  // terminalDAM->set_armature(armature);
  runningModel =
      boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
  terminalModel =
      boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(0.));
}

} // namespace benchmark
} // namespace crocoddyl

#endif // CROCODDYL_ARM_FACTORY_HPP_
