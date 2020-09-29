///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_ARM_COLLISION_FACTORY_HPP_
#define CROCODDYL_ARM_COLLISION_FACTORY_HPP_

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/fcl.hpp>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <example-robot-data/path.hpp>

#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/pair-collisions.hpp"
#include "crocoddyl/multibody/costs/control.hpp"

namespace crocoddyl {
namespace benchmark {

template <typename Scalar>
void build_arm_action_models_w_collision(boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& runningModel,
                                         boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& terminalModel) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::FramePlacementTpl<Scalar> FramePlacement;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelFramePlacementTpl<Scalar> CostModelFramePlacement;
  typedef typename crocoddyl::CostModelStateTpl<Scalar> CostModelState;
  typedef typename crocoddyl::CostModelControlTpl<Scalar> CostModelControl;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::ActuationModelFullTpl<Scalar> ActuationModelFull;
  typedef typename crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar> DifferentialActionModelFreeFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar> IntegratedActionModelEuler;

  // because urdf is not supported with all scalar types.
  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", modeld);
  pinocchio::srdf::loadReferenceConfigurations(modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                               false);
  
  pinocchio::ModelTpl<Scalar> model_full(modeld.cast<Scalar>()), model;
  std::vector<pinocchio::JointIndex> locked_joints;
  locked_joints.push_back(5);
  locked_joints.push_back(6);
  locked_joints.push_back(7);
  pinocchio::buildReducedModel(model_full, locked_joints, VectorXs::Zero(model_full.nq), model);


  pinocchio::GeometryModel geomModel;

  std::string link_name = "arm_left_4_joint";
  double capsule_length = 0.45;
  Eigen::Vector3d pos_body(-0.025,0,-.225);
  pinocchio::FrameIndex pin_link_id = model.getFrameId(link_name);
  pinocchio::JointIndex pin_joint_id = model.getJointId(link_name); //arm_left_4_joint in pinocchio

  pinocchio::GeomIndex ig_arm = geomModel.addGeometryObject(pinocchio::GeometryObject("simple_arm", pin_link_id, model.frames[model.getFrameId("arm_left_4_link")].parent, boost::shared_ptr<hpp::fcl::Capsule>(new hpp::fcl::Capsule(0, capsule_length)), pinocchio::SE3(Eigen::Matrix3d::Identity(),pos_body)),model);


  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::Vector3d) box_poses, box_sizes;
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::Vector2d) capsule_sizes;
  std::vector<double> obs_radius;
  //Table, green cube, yellow cube
  box_poses.push_back(Eigen::Vector3d(-0.3,-0.1,0.75));
  box_poses.push_back(Eigen::Vector3d(-0.45, -0.3, 0.7925 + 0.01));
  box_poses.push_back(Eigen::Vector3d(-0.45, -0.3, 0.8925 + 0.01));


  box_sizes.push_back(Eigen::Vector3d(.25, .3, .01));
  box_sizes.push_back(Eigen::Vector3d(0.05,0.05, 0.05));
  box_sizes.push_back(Eigen::Vector3d(0.05,0.05, 0.05));

  capsule_sizes.push_back(Eigen::Vector2d(.3, .02));
  capsule_sizes.push_back(Eigen::Vector2d(0.05, 0.1));
  capsule_sizes.push_back(Eigen::Vector2d(0.05, 0.1));
  
  obs_radius.push_back(0.005);
  obs_radius.push_back(0.05);
  obs_radius.push_back(0.05);

  double add_threshold = 0.03;
  double RADIUS = 0.09;
  
  int num_obs = box_poses.size();
  // Add obstacles in the world
  for(int i=0;i<num_obs;++i) {
    pinocchio::GeomIndex ig_obs = geomModel.addGeometryObject(
       pinocchio::GeometryObject("simple_obs"+std::to_string(i),
                                 model.getFrameId("universe"),
                                 model.frames[model.getFrameId("universe")].parent,
                                 boost::shared_ptr<hpp::fcl::Capsule>(new hpp::fcl::Capsule(capsule_sizes[i](0), capsule_sizes[i](1))),
                                 pinocchio::SE3(Eigen::Matrix3d::Identity(), box_poses[i])),
       model);    
    geomModel.addCollisionPair(pinocchio::CollisionPair(ig_arm,ig_obs));
  }

  
  
  boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
          boost::make_shared<pinocchio::ModelTpl<Scalar> >(model));

  // We define an actuation model
  boost::shared_ptr<ActuationModelFull> actuation = boost::make_shared<ActuationModelFull>(state);
  

  FramePlacement Mref(model.getFrameId("gripper_left_joint"),
                      pinocchio::SE3Tpl<Scalar>(Matrix3s::Identity(), Vector3s(Scalar(0), Scalar(0), Scalar(.4))));
  boost::shared_ptr<CostModelAbstract> goalTrackingCost = boost::make_shared<CostModelFramePlacement>(state, Mref);
  boost::shared_ptr<CostModelAbstract> xRegCost = boost::make_shared<CostModelState>(state);
  boost::shared_ptr<CostModelAbstract> uRegCost = boost::make_shared<CostModelControl>(state);


  std::vector<boost::shared_ptr<CostModelAbstract> > obstacleCosts;


  for(int i=0;i<box_sizes.size(); ++i) {
    obstacleCosts.push_back(boost::make_shared<CostModelPairCollisions>(
          state, RADIUS+obs_radius[i]+add_threshold,
          actuation->get_nu(),
          boost::shared_ptr<pinocchio::GeometryModel>(boost::make_shared<pinocchio::GeometryModel>(geomModel)), i, pin_joint_id));
  }

  

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<CostModelSum> runningCostModel = boost::make_shared<CostModelSum>(state);
  boost::shared_ptr<CostModelSum> terminalCostModel = boost::make_shared<CostModelSum>(state);

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));

  for(int i=0;i<box_sizes.size();++i) {
    runningCostModel->addCost("obstacle"+std::to_string(i), obstacleCosts[i], Scalar(1e4));
  }
  
  terminalCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  for(int i=0;i<box_sizes.size();++i) {
    terminalCostModel->addCost("obstacle"+std::to_string(i), obstacleCosts[i], Scalar(1e4));
  }  

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<DifferentialActionModelFreeFwdDynamics> runningDAM =
      boost::make_shared<DifferentialActionModelFreeFwdDynamics>(state, actuation, runningCostModel);

  // VectorXs armature(state->get_nq());
  // armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  // runningDAM->set_armature(armature);
  // terminalDAM->set_armature(armature);
  runningModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
  terminalModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(0.));
}
  
}  // namespace benchmark
}  // namespace crocoddyl

#endif  // CROCODDYL_ARM_COLLISION_FACTORY_HPP_
