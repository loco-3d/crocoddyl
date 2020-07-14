///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/container/aligned-vector.hpp>

#include <example-robot-data/path.hpp>

#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/codegen/action-base.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"

#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"

#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"

#include "factory/solver.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

/// \brief Changing the environment variables in a autodiff model. This function needs to be passed to the
/// ActionModelCodeGen in order to make the calc and calcdiff be dependent on
///        some parameter of the action model (like a cost reference). Inside the function definition, set the
///        env_vector where you want it to be defined inside ad_model.
/// \param[in,out] ad_model    the ActionModelCodeGen that needs to be recorded
/// \param[in]     env_vector  the environment vector which would be set in ad_model.
template <typename Scalar>
void change_env(boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > ad_model,
                const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& env_vector) {
  crocoddyl::IntegratedActionModelEulerTpl<Scalar>* m =
      static_cast<crocoddyl::IntegratedActionModelEulerTpl<Scalar>*>(ad_model.get());
  crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>* md =
      static_cast<crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>*>(m->get_differential().get());
  crocoddyl::FrameTranslationTpl<Scalar> Tref(md->get_pinocchio().getFrameId("gripper_left_joint"), env_vector);
  md->get_costs()->get_costs().find("gripperTrans")->second->cost->set_reference(Tref);
}

template <typename Scalar>
const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > build_arm_action_model() {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::FramePlacementTpl<Scalar> FramePlacement;
  typedef typename crocoddyl::FrameTranslationTpl<Scalar> FrameTranslation;
  typedef typename crocoddyl::FrameRotationTpl<Scalar> FrameRotation;
  typedef typename crocoddyl::FrameMotionTpl<Scalar> FrameMotion;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelFramePlacementTpl<Scalar> CostModelFramePlacement;
  typedef typename crocoddyl::CostModelFrameTranslationTpl<Scalar> CostModelFrameTranslation;
  typedef typename crocoddyl::CostModelFrameRotationTpl<Scalar> CostModelFrameRotation;
  typedef typename crocoddyl::CostModelFrameVelocityTpl<Scalar> CostModelFrameVelocity;
  typedef typename crocoddyl::CostModelStateTpl<Scalar> CostModelState;
  typedef typename crocoddyl::CostModelControlTpl<Scalar> CostModelControl;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef typename crocoddyl::ActuationModelFullTpl<Scalar> ActuationModelFull;
  typedef typename crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar> DifferentialActionModelFreeFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar> IntegratedActionModelEuler;

  typedef typename crocoddyl::ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef typename crocoddyl::ActivationModelQuadraticBarrierTpl<Scalar> ActivationModelQuadraticBarrier;
  typedef typename crocoddyl::ActivationModelWeightedQuadraticBarrierTpl<Scalar>
      ActivationModelWeightedQuadraticBarrier;

  // because urdf is not supported with all scalar types.
  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", modeld);
  pinocchio::srdf::loadReferenceConfigurations(modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                               false);

  pinocchio::ModelTpl<Scalar> model_full(modeld.cast<Scalar>()), model;
  std::vector<pinocchio::JointIndex> locked_joints{5, 6, 7};
  pinocchio::buildReducedModel(model_full, locked_joints, VectorXs::Zero(model_full.nq), model);

  boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
          boost::make_shared<pinocchio::ModelTpl<Scalar> >(model));

  FramePlacement Mref(model.getFrameId("gripper_left_joint"),
                      pinocchio::SE3Tpl<Scalar>(Matrix3s::Identity(), Vector3s(Scalar(0), Scalar(0), Scalar(.4))));
  FrameTranslation Tref(model.getFrameId("gripper_left_joint"), Vector3s(Scalar(0), Scalar(0), Scalar(.4)));
  FrameRotation Rref(model.getFrameId("gripper_left_joint"), Matrix3s::Identity());
  FrameMotion Vref(model.getFrameId("gripper_left_joint"),
                   pinocchio::MotionTpl<Scalar>(Vector3s(Scalar(0), Scalar(0), Scalar(.4)),
                                                Vector3s(Scalar(0), Scalar(0), Scalar(.4))));
  boost::shared_ptr<CostModelAbstract> goalTrackingCost = boost::make_shared<CostModelFramePlacement>(state, Mref);
  boost::shared_ptr<CostModelAbstract> goalTranslationCost =
      boost::make_shared<CostModelFrameTranslation>(state, Tref);
  boost::shared_ptr<CostModelAbstract> goalRotationCost = boost::make_shared<CostModelFrameRotation>(state, Rref);
  boost::shared_ptr<CostModelAbstract> goalVelocityCost = boost::make_shared<CostModelFrameVelocity>(state, Vref);
  boost::shared_ptr<CostModelAbstract> xRegCost = boost::make_shared<CostModelState>(state);
  boost::shared_ptr<CostModelAbstract> uRegCost = boost::make_shared<CostModelControl>(state);

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<CostModelSum> runningCostModel = boost::make_shared<CostModelSum>(state);

  VectorXs lowlim = (model.lowerPositionLimit);
  VectorXs uplim = (model.upperPositionLimit);
  VectorXs xlb(model.nq + model.nv), xub(model.nq + model.nv);
  xlb << lowlim, -VectorXs::Ones(model.nv);
  xub << uplim, VectorXs::Ones(model.nv);

  // xlb.tail(model.nv) *= Scalar(-1) * std::numeric_limits<Scalar>::max();
  // xub.tail(model.nv) *= std::numeric_limits<Scalar>::max();

  VectorXs xweights(model.nv + model.nv);
  xweights.head(model.nv).fill(Scalar(10.));
  xweights.tail(model.nv).fill(Scalar(100.));

  ActivationBounds bounds(xlb, xub);
  boost::shared_ptr<ActivationModelQuadraticBarrier> activation_bounded =
      boost::make_shared<ActivationModelQuadraticBarrier>(bounds);
  boost::shared_ptr<ActivationModelWeightedQuadraticBarrier> weighted_activation_bounded =
      boost::make_shared<ActivationModelWeightedQuadraticBarrier>(bounds, xweights);

  boost::shared_ptr<CostModelAbstract> jointLimitCost = boost::make_shared<CostModelState>(state, activation_bounded);

  boost::shared_ptr<CostModelAbstract> jointLimitCost2 =
      boost::make_shared<CostModelState>(state, weighted_activation_bounded);

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("gripperTrans", goalTranslationCost, Scalar(1));
  runningCostModel->addCost("gripperRot", goalRotationCost, Scalar(1));
  runningCostModel->addCost("gripperVel", goalVelocityCost, Scalar(1));
  runningCostModel->addCost("jointLim", jointLimitCost, Scalar(1e3));
  runningCostModel->addCost("jointLim2", jointLimitCost2, Scalar(1e3));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));

  // We define an actuation model
  boost::shared_ptr<ActuationModelFull> actuation = boost::make_shared<ActuationModelFull>(state);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<DifferentialActionModelFreeFwdDynamics> runningDAM =
      boost::make_shared<DifferentialActionModelFreeFwdDynamics>(state, actuation, runningCostModel);

  // VectorXs armature(state->get_nq());
  // armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  // runningDAM->set_armature(armature);
  // terminalDAM->set_armature(armature);
  boost::shared_ptr<ActionModelAbstract> runningModel =
      boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
  return runningModel;
}

template <typename Scalar>
const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > build_bipedal_action_model() {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector2s Vector2s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector6s Vector6s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::FramePlacementTpl<Scalar> FramePlacement;
  typedef typename crocoddyl::FrameTranslationTpl<Scalar> FrameTranslation;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelFramePlacementTpl<Scalar> CostModelFramePlacement;
  typedef typename crocoddyl::CostModelStateTpl<Scalar> CostModelState;
  typedef typename crocoddyl::CostModelControlTpl<Scalar> CostModelControl;
  typedef typename crocoddyl::CostModelCoMPositionTpl<Scalar> CostModelCoMPosition;
  typedef typename crocoddyl::CostModelContactForceTpl<Scalar> CostModelContactForce;
  typedef typename crocoddyl::CostModelCentroidalMomentumTpl<Scalar> CostModelCentroidalMomentum;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::FrameForceTpl<Scalar> FrameForce;
  typedef typename crocoddyl::ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef typename crocoddyl::ActuationModelFloatingBaseTpl<Scalar> ActuationModelFloatingBase;
  typedef typename crocoddyl::DifferentialActionModelContactFwdDynamicsTpl<Scalar>
      DifferentialActionModelContactFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar> IntegratedActionModelEuler;

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

  boost::shared_ptr<CostModelAbstract> goalTrackingCost =
      boost::make_shared<CostModelFramePlacement>(state, Mref, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> centroidalCost =
      boost::make_shared<CostModelCentroidalMomentum>(state, Vector6s::Zero(), actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> comCost =
      boost::make_shared<CostModelCoMPosition>(state, Vector3s::Zero(), actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> contactForceCost = boost::make_shared<CostModelContactForce>(
      state, FrameForce(model.getFrameId(RF), pinocchio::ForceTpl<Scalar>::Zero()), 6, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> xRegCost = boost::make_shared<CostModelState>(state, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> uRegCost = boost::make_shared<CostModelControl>(state, actuation->get_nu());

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<CostModelSum> runningCostModel = boost::make_shared<CostModelSum>(state, actuation->get_nu());

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));
  runningCostModel->addCost("contactforce", contactForceCost, Scalar(1e-4));
  runningCostModel->addCost("comcost", comCost, Scalar(1e-4));
  runningCostModel->addCost("centroidal", centroidalCost, Scalar(1e-4));

  boost::shared_ptr<ContactModelMultiple> contact_models =
      boost::make_shared<ContactModelMultiple>(state, actuation->get_nu());

  FramePlacement xref(model.getFrameId(RF), pinocchio::SE3Tpl<Scalar>::Identity());
  boost::shared_ptr<ContactModelAbstract> support_contact_model6D =
      boost::make_shared<ContactModel6D>(state, xref, actuation->get_nu(), Vector2s(Scalar(0.), Scalar(50.)));
  contact_models->addContact(model.frames[model.getFrameId(RF)].name + "_contact", support_contact_model6D);

  FrameTranslation x2ref(model.getFrameId(LF), Vector3s::Zero());
  boost::shared_ptr<ContactModelAbstract> support_contact_model3D =
      boost::make_shared<ContactModel3D>(state, x2ref, actuation->get_nu(), Vector2s(Scalar(0.), Scalar(50.)));
  contact_models->addContact(model.frames[model.getFrameId(LF)].name + "_contact", support_contact_model3D);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> runningDAM =
      boost::make_shared<DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                    runningCostModel);

  // VectorXs armature(state->get_nq());
  // armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  // runningDAM->set_armature(armature);
  // terminalDAM->set_armature(armature);
  boost::shared_ptr<ActionModelAbstract> runningModel =
      boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
  return runningModel;
}

void test_codegen_4DoFArm() {
  typedef double Scalar;
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > runningModelD = build_arm_action_model<Scalar>();
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar> > runningModelAD = build_arm_action_model<ADScalar>();

  // The definition of the ActionModelCodeGen takes the size of the environment variable, and the function setting the
  // environment variable as arguments.
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > runningModelCG =
      boost::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar> >(runningModelAD, runningModelD,
                                                                    "pyrene_arm_running", 3, change_env<ADScalar>);

  // Check that code-generated action model is the same as original.
  /**************************************************************************/
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > runningDataCG = runningModelCG->createData();
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > runningDataD = runningModelD->createData();

  // Change cost reference ********************************************************/
  const Vector3s new_ref(Vector3s::Random());
  crocoddyl::ActionModelCodeGenTpl<Scalar>* rmcg =
      static_cast<crocoddyl::ActionModelCodeGenTpl<Scalar>*>(runningModelCG.get());
  rmcg->set_env(runningDataCG, new_ref);
  crocoddyl::IntegratedActionModelEulerTpl<Scalar>* m =
      static_cast<crocoddyl::IntegratedActionModelEulerTpl<Scalar>*>(runningModelD.get());
  crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>* md =
      static_cast<crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar>*>(m->get_differential().get());
  crocoddyl::FrameTranslationTpl<Scalar> Tref(md->get_pinocchio().getFrameId("gripper_left_joint"), new_ref);
  md->get_costs()->get_costs().find("gripperTrans")->second->cost->set_reference(Tref);
  /*************************************************************/

  VectorXs x_rand = runningModelCG->get_state()->rand();
  VectorXs u_rand = VectorXs::Random(runningModelCG->get_nu());
  runningModelD->calc(runningDataD, x_rand, u_rand);
  runningModelD->calcDiff(runningDataD, x_rand, u_rand);
  runningModelCG->calc(runningDataCG, x_rand, u_rand);
  runningModelCG->calcDiff(runningDataCG, x_rand, u_rand);

  BOOST_CHECK(runningDataCG->xnext.isApprox(runningDataD->xnext));
  BOOST_CHECK_CLOSE(runningDataCG->cost, runningDataD->cost, Scalar(1e-10));
  BOOST_CHECK(runningDataCG->Lx.isApprox(runningDataD->Lx));
  BOOST_CHECK(runningDataCG->Lu.isApprox(runningDataD->Lu));
  BOOST_CHECK(runningDataCG->Lxx.isApprox(runningDataD->Lxx));
  BOOST_CHECK(runningDataCG->Lxu.isApprox(runningDataD->Lxu));
  BOOST_CHECK(runningDataCG->Luu.isApprox(runningDataD->Luu));
  BOOST_CHECK(runningDataCG->Fx.isApprox(runningDataD->Fx));
  BOOST_CHECK(runningDataCG->Fu.isApprox(runningDataD->Fu));
}

void test_codegen_bipedal() {
  typedef double Scalar;
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > runningModelD = build_bipedal_action_model<Scalar>();
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar> > runningModelAD =
      build_bipedal_action_model<ADScalar>();

  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > runningModelCG =
      boost::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar> >(runningModelAD, runningModelD, "pyrene_biped");

  // Check that code-generated action model is the same as original.
  /**************************************************************************/
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > runningDataCG = runningModelCG->createData();
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > runningDataD = runningModelD->createData();
  VectorXs x_rand = runningModelCG->get_state()->rand();
  VectorXs u_rand = VectorXs::Random(runningModelCG->get_nu());
  runningModelD->calc(runningDataD, x_rand, u_rand);
  runningModelD->calcDiff(runningDataD, x_rand, u_rand);
  runningModelCG->calc(runningDataCG, x_rand, u_rand);
  runningModelCG->calcDiff(runningDataCG, x_rand, u_rand);

  BOOST_CHECK(runningDataCG->xnext.isApprox(runningDataD->xnext));
  BOOST_CHECK_CLOSE(runningDataCG->cost, runningDataD->cost, Scalar(1e-10));
  BOOST_CHECK(runningDataCG->Lx.isApprox(runningDataD->Lx));
  BOOST_CHECK(runningDataCG->Lu.isApprox(runningDataD->Lu));
  BOOST_CHECK(runningDataCG->Lxx.isApprox(runningDataD->Lxx));
  BOOST_CHECK(runningDataCG->Lxu.isApprox(runningDataD->Lxu));
  BOOST_CHECK(runningDataCG->Luu.isApprox(runningDataD->Luu));
  BOOST_CHECK(runningDataCG->Fx.isApprox(runningDataD->Fx));
  BOOST_CHECK(runningDataCG->Fu.isApprox(runningDataD->Fu));
}

bool init_function() {
  const std::string test_name = "test_codegen";
  test_suite* ts = BOOST_TEST_SUITE(test_name);
  ts->add(BOOST_TEST_CASE(&test_codegen_4DoFArm));
  ts->add(BOOST_TEST_CASE(&test_codegen_bipedal));
  framework::master_test_suite().add(ts);

  return true;
}

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
