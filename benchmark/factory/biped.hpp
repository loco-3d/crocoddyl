///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_BIPED_FACTORY_HPP_
#define CROCODDYL_BIPED_FACTORY_HPP_

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <example-robot-data/path.hpp>

#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"

namespace crocoddyl {
namespace benchmark {

template <typename Scalar>
void build_biped_action_models(boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& runningModel,
                               boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& terminalModel) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector2s Vector2s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::FramePlacementTpl<Scalar> FramePlacement;
  typedef typename crocoddyl::FrameTranslationTpl<Scalar> FrameTranslation;
  typedef typename crocoddyl::DifferentialActionModelContactFwdDynamicsTpl<Scalar>
      DifferentialActionModelContactFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar> IntegratedActionModelEuler;
  typedef typename crocoddyl::ActuationModelFloatingBaseTpl<Scalar> ActuationModelFloatingBase;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::ContactModelAbstractTpl<Scalar> ContactModelAbstract;
  typedef typename crocoddyl::CostModelFramePlacementTpl<Scalar> CostModelFramePlacement;
  typedef typename crocoddyl::CostModelCoMPositionTpl<Scalar> CostModelCoMPosition;
  typedef typename crocoddyl::CostModelStateTpl<Scalar> CostModelState;
  typedef typename crocoddyl::CostModelControlTpl<Scalar> CostModelControl;
  typedef typename crocoddyl::ContactModel6DTpl<Scalar> ContactModel6D;
  typedef typename crocoddyl::ContactModel3DTpl<Scalar> ContactModel3D;

  const std::string RF = "leg_right_6_joint";
  const std::string LF = "leg_left_6_joint";

  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf",
                              pinocchio::JointModelFreeFlyer(), modeld);
  modeld.lowerPositionLimit.head<7>().array() = -1;
  modeld.upperPositionLimit.head<7>().array() = 1.;
  pinocchio::srdf::loadReferenceConfigurations(modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                               false);

  pinocchio::ModelTpl<Scalar> model(modeld.cast<Scalar>());
  boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
          boost::make_shared<pinocchio::ModelTpl<Scalar> >(model));

  boost::shared_ptr<ActuationModelFloatingBase> actuation = boost::make_shared<ActuationModelFloatingBase>(state);

  FramePlacement Mref(model.getFrameId("arm_right_7_joint"),
                      pinocchio::SE3Tpl<Scalar>(Matrix3s::Identity(), Vector3s(Scalar(.0), Scalar(.0), Scalar(.4))));

  boost::shared_ptr<CostModelAbstract> comCost =
      boost::make_shared<CostModelCoMPosition>(state, Vector3s::Zero(), actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> goalTrackingCost =
      boost::make_shared<CostModelFramePlacement>(state, Mref, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> xRegCost = boost::make_shared<CostModelState>(state, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> uRegCost = boost::make_shared<CostModelControl>(state, actuation->get_nu());

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<CostModelSum> runningCostModel = boost::make_shared<CostModelSum>(state, actuation->get_nu());
  boost::shared_ptr<CostModelSum> terminalCostModel = boost::make_shared<CostModelSum>(state, actuation->get_nu());

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  //   runningCostModel->addCost("comPos", comCost, Scalar(1e-7));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));
  terminalCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));

  boost::shared_ptr<ContactModelMultiple> contact_models =
      boost::make_shared<ContactModelMultiple>(state, actuation->get_nu());

  FramePlacement RFref(model.getFrameId(RF), pinocchio::SE3Tpl<Scalar>::Identity());
  boost::shared_ptr<ContactModelAbstract> support_contact_RF =
      boost::make_shared<ContactModel6D>(state, RFref, actuation->get_nu(), Vector2s(Scalar(0.), Scalar(50.)));
  contact_models->addContact(model.frames[model.getFrameId(RF)].name, support_contact_RF);

  FrameTranslation LFref(model.getFrameId(LF), Eigen::Vector3d::Zero());
  boost::shared_ptr<ContactModelAbstract> support_contact_LF =
      boost::make_shared<ContactModel3D>(state, LFref, actuation->get_nu(), Vector2s(Scalar(0.), Scalar(50.)));
  contact_models->addContact(model.frames[model.getFrameId(LF)].name, support_contact_LF);

  // Next, we need to create an action model for running and terminal nodes
  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> runningDAM =
      boost::make_shared<DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                    runningCostModel);
  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> terminalDAM =
      boost::make_shared<DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                    terminalCostModel);

  runningModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
  terminalModel = boost::make_shared<IntegratedActionModelEuler>(terminalDAM, Scalar(1e-3));
}

}  // namespace benchmark
}  // namespace crocoddyl

#endif  // CROCODDYL_BIPED_FACTORY_HPP_