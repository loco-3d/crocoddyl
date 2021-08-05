///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff_action.hpp"
#include "cost.hpp"
#include "contact.hpp"

#include <pinocchio/parsers/sdf.hpp>

#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<DifferentialActionModelTypes::Type> DifferentialActionModelTypes::all(
    DifferentialActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, DifferentialActionModelTypes::Type type) {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      os << "DifferentialActionModelLQR";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      os << "DifferentialActionModelLQRDriftFree";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      os << "DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_TalosArm:
      os << "DifferentialActionModelContactFwdDynamics_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_HyQ:
      os << "DifferentialActionModelContactFwdDynamics_HyQ";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_Talos:
      os << "DifferentialActionModelContactFwdDynamics_Talos";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics2_Cassie:
      os << "DifferentialActionModelContactFwdDynamics2_Cassie";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_HyQ";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
      os << "DifferentialActionModelContactFwdDynamicsWithFriction_Talos";
      break;
    case DifferentialActionModelTypes::NbDifferentialActionModelTypes:
      os << "NbDifferentialActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

DifferentialActionModelFactory::DifferentialActionModelFactory() {}
DifferentialActionModelFactory::~DifferentialActionModelFactory() {}

boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> DifferentialActionModelFactory::create(
    DifferentialActionModelTypes::Type type) const {
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action;
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      action = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      action = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(40, 40, true);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm:
      action =
          create_freeFwdDynamics(StateModelTypes::StateMultibody_TalosArm, ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      action = create_freeFwdDynamics(StateModelTypes::StateMultibody_TalosArm,
                                      ActuationModelTypes::ActuationModelSquashingFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_TalosArm:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_TalosArm,
                                         ActuationModelTypes::ActuationModelFull, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_HyQ:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_HyQ,
                                         ActuationModelTypes::ActuationModelFloatingBase, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics_Talos:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos,
                                         ActuationModelTypes::ActuationModelFloatingBase, false);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics2_Cassie:
      action = create_cassieContactFwdDynamics2();
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_TalosArm:
      action =
          create_contactFwdDynamics(StateModelTypes::StateMultibody_TalosArm, ActuationModelTypes::ActuationModelFull);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_HyQ:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_HyQ,
                                         ActuationModelTypes::ActuationModelFloatingBase);
      break;
    case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamicsWithFriction_Talos:
      action = create_contactFwdDynamics(StateModelTypes::StateMultibody_Talos,
                                         ActuationModelTypes::ActuationModelFloatingBase);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
      break;
  }
  return action;
}

boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics>
DifferentialActionModelFactory::create_freeFwdDynamics(StateModelTypes::Type state_type,
                                                       ActuationModelTypes::Type actuation_type) const {
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelResidualState, state_type,
                                          ActivationModelTypes::ActivationModelQuad),
                1.);
  cost->addCost("control",
                CostModelFactory().create(CostModelTypes::CostModelResidualControl, state_type,
                                          ActivationModelTypes::ActivationModelQuad),
                1.);
  cost->addCost("frame",
                CostModelFactory().create(CostModelTypes::CostModelResidualFramePlacement, state_type,
                                          ActivationModelTypes::ActivationModelQuad),
                1.);
  action = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, cost);
  return action;
}

boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
DifferentialActionModelFactory::create_contactFwdDynamics(StateModelTypes::Type state_type,
                                                          ActuationModelTypes::Type actuation_type,
                                                          bool with_friction) const {
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  contact = boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  crocoddyl::FrictionCone cone(R, 0.8, 4, false);
  crocoddyl::ActivationBounds bounds(cone.get_lb(), cone.get_ub());
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation =
      boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm:
      contact->addContact("lf", boost::make_shared<crocoddyl::ContactModel3D>(
                                    state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"),
                                    Eigen::Vector3d::Zero(), actuation->get_nu()));
      if (with_friction) {
        cost->addCost("lf_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("gripper_left_fingertip_1_link"), cone,
                              actuation->get_nu())),
                      0.1);
      }
      break;
    case StateModelTypes::StateMultibody_HyQ:
      contact->addContact(
          "lf", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "lf_foot",
                                             actuation->get_nu()));
      contact->addContact(
          "rf", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "rf_foot",
                                             actuation->get_nu()));
      contact->addContact(
          "lh", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "lh_foot",
                                             actuation->get_nu()));
      contact->addContact(
          "rh", ContactModelFactory().create(ContactModelTypes::ContactModel3D, PinocchioModelTypes::HyQ, "rh_foot",
                                             actuation->get_nu()));
      if (with_friction) {
        cost->addCost("lf_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("lf_foot"), cone, actuation->get_nu())),
                      0.1);
        cost->addCost("rf_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("rf_foot"), cone, actuation->get_nu())),
                      0.1);
        cost->addCost("lh_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("lh_foot"), cone, actuation->get_nu())),
                      0.1);
        cost->addCost("rh_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("rh_foot"), cone, actuation->get_nu())),
                      0.1);
      }
      break;
    case StateModelTypes::StateMultibody_Talos:
      contact->addContact("lf",
                          ContactModelFactory().create(ContactModelTypes::ContactModel6D, PinocchioModelTypes::Talos,
                                                       "left_sole_link", actuation->get_nu()));
      contact->addContact("rf",
                          ContactModelFactory().create(ContactModelTypes::ContactModel6D, PinocchioModelTypes::Talos,
                                                       "right_sole_link", actuation->get_nu()));
      if (with_friction) {
        cost->addCost("lf_cone",
                      boost::make_shared<crocoddyl::CostModelResidual>(
                          state, activation,
                          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                              state, state->get_pinocchio()->getFrameId("left_sole_link"), cone, actuation->get_nu())),
                      0.1);
        cost->addCost(
            "rf_cone",
            boost::make_shared<crocoddyl::CostModelResidual>(
                state, activation,
                boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, state->get_pinocchio()->getFrameId("right_sole_link"), cone, actuation->get_nu())),
            0.1);
      }
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  cost->addCost("state",
                CostModelFactory().create(CostModelTypes::CostModelResidualState, state_type,
                                          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                0.1);
  cost->addCost("control",
                CostModelFactory().create(CostModelTypes::CostModelResidualControl, state_type,
                                          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
                0.1);
  action = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state, actuation, contact, cost,
                                                                                    0., true);
  return action;
}



boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics2>
DifferentialActionModelFactory::create_cassieContactFwdDynamics2() const {
  
  const std::string filename = "/home/rbudhira/devel/src/misc/cassie-gazebo-sim/cassie/cassie_v2.sdf";
  
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::Model model;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidContactModel) contact_models_empty;
  pinocchio::sdf::buildModel(filename, root_joint, model, contact_models);

  for(int i=0; i<contact_models.size(); ++i) {
    std::cerr<<i<<"joint 1 name"<<model.names[contact_models[i].joint1_id]<<std::endl;
    std::cerr<<i<<"joint 2 name"<<model.names[contact_models[i].joint2_id]<<std::endl;
  }
  
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics2> action;
;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;

  boost::shared_ptr<crocoddyl::StateMultibody> state =
    boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));

  std::vector<std::string> actuated_joints {"left-roll-op",
      "left-yaw-op",
      "left-pitch-op",
      "left-knee-op",
      "left-knee-shin-joint",
      "left-shin-tarsus-joint",
      "left-foot-op",
      "right-roll-op",
      "right-yaw-op",
      "right-pitch-op",
      "right-knee-op",
      "left-knee-shin-joint",
      "left-shin-tarsus-joint",
      "right-foot-op"};

  std::vector<pinocchio::JointIndex> actuated_dofs;
  for(std::size_t j=0;j<actuated_joints.size(); j++) {
    actuated_dofs.push_back(model.idx_vs[model.getJointId(actuated_joints[j])]);
  }
  
  Eigen::MatrixXd actuation_matrix(model.nv, actuated_dofs.size());
  actuation_matrix.setZero();
  for(std::size_t j=0;j<actuated_dofs.size(); j++) {
    actuation_matrix(actuated_dofs[j], j) = 1.;
  }

  Eigen::MatrixXd spring_actuation_matrix(model.nv, model.nv);
  Eigen::MatrixXd damping_actuation_matrix(model.nv, model.nv);

  spring_actuation_matrix.setZero();
  damping_actuation_matrix.setZero();

  Eigen::VectorXd x_base(model.nq + model.nv);
  x_base << 0.   ,  0.   ,  1.104,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,
    0.   ,  0.298,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.398,
    1.398, -1.398,  0.   ,  0.   ,  0.298,  0.   ,  0.   ,  0.   ,
    0.   ,  0.   , -1.398,  1.398, -1.398, Eigen::VectorXd::Zero(28);
  // actuation =
  //   boost::make_shared<crocoddyl::ActuationModelFloatingBaseWithPassiveJoints>(
  //                                                      state,
  //                                                      actuation_matrix,
  //                                                      spring_actuation_matrix,
  //                                                      damping_actuation_matrix);

  actuation =
    boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);
  
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  
  cost->addCost("state", boost::make_shared<crocoddyl::CostModelState>(
                          state,
                          boost::make_shared<crocoddyl::ActivationModelQuad>(state->get_ndx()),
			  x_base,
                          actuation->get_nu()),
                0.1);

  cost->addCost("control", boost::make_shared<crocoddyl::CostModelControl>(
                           state,
                           boost::make_shared<crocoddyl::ActivationModelQuad>(actuation->get_nu()),
                           Eigen::VectorXd::Random(actuation->get_nu())),
                0.1);
  
  cost->addCost("com", boost::make_shared<crocoddyl::CostModelCoMPosition>(
                          state,
                          boost::make_shared<crocoddyl::ActivationModelQuad>(3),
                          Eigen::Vector3d::Random(),
                          actuation->get_nu()),
                0.1);
  
  action = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics2>(state, actuation, contact_models, cost, 1);
  //action = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics2>(state, actuation, contact_models_empty, cost, 1e-5);
  return action;
}
  
}  // namespace unittest
}  // namespace crocoddyl
