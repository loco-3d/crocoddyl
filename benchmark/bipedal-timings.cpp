///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <example-robot-data/path.hpp>

#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/codegen/action-base.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"

#include "crocoddyl/core/mathbase.hpp"

#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#define SMOOTH(s) for (size_t _smooth = 0; _smooth < s; ++_smooth)

#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / (vec.size() - 1))
#define AVG(vec) (vec.mean())

int main(int argc, char* argv[]) {
  unsigned int N = 100;  // number of nodes
  unsigned int T = 5e4;  // number of trials
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  /**************************DOUBLE**********************/
  /**************************DOUBLE**********************/
  /**************************DOUBLE**********************/
  pinocchio::Model model;

  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf",
                              pinocchio::JointModelFreeFlyer(), model);
  model.lowerPositionLimit.head<7>().array() = -1;
  model.upperPositionLimit.head<7>().array() = 1.;

  pinocchio::srdf::loadReferenceConfigurations(model, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                               false);
  const std::string RF = "leg_right_6_joint";
  const std::string LF = "leg_left_6_joint";

  /*************************PINOCCHIO MODEL**************/

  /************************* SETUP ***********************/
  crocoddyl::Timer timer;
  std::cout << "NQ: " << model.nq << std::endl;

  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation =
      boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);

  Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
  Eigen::VectorXd x0(state->get_nx());
  x0 << q0, Eigen::VectorXd::Random(state->get_nv());

  crocoddyl::FramePlacement Mref(model.getFrameId("arm_right_7_joint"),
                                 pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.0, .0, .4)));

  boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
      boost::make_shared<crocoddyl::CostModelFramePlacement>(state, Mref, actuation->get_nu());

  boost::shared_ptr<crocoddyl::CostModelAbstract> xRegCost =
      boost::make_shared<crocoddyl::CostModelState>(state, actuation->get_nu());
  boost::shared_ptr<crocoddyl::CostModelAbstract> uRegCost =
      boost::make_shared<crocoddyl::CostModelControl>(state, actuation->get_nu());

  boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel =
      boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel =
      boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  runningCostModel->addCost("gripperPose", goalTrackingCost, 1);
  runningCostModel->addCost("xReg", xRegCost, 1e-4);
  runningCostModel->addCost("uReg", uRegCost, 1e-4);
  terminalCostModel->addCost("gripperPose", goalTrackingCost, 1);

  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact_models =
      boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());

  crocoddyl::FramePlacement xref(model.getFrameId(RF), pinocchio::SE3::Identity());
  boost::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model6D =
      boost::make_shared<crocoddyl::ContactModel6D>(state, xref, actuation->get_nu(), Eigen::Vector2d(0., 50.));
  contact_models->addContact(model.frames[model.getFrameId(RF)].name + "_contact", support_contact_model6D);

  crocoddyl::FrameTranslation x2ref(model.getFrameId(LF), Eigen::Vector3d::Zero());
  boost::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model3D =
      boost::make_shared<crocoddyl::ContactModel3D>(state, x2ref, actuation->get_nu(), Eigen::Vector2d(0., 50.));
  contact_models->addContact(model.frames[model.getFrameId(LF)].name + "_contact", support_contact_model3D);

  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> runningDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                               runningCostModel);

  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminalDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                               terminalCostModel);

  boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(runningDAM, 1e-3);
  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(terminalDAM, 1e-3);

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(N, runningModel);

  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, terminalModel);
  std::vector<Eigen::VectorXd> xs(N + 1, x0);

  /***************************************************************/

  boost::shared_ptr<crocoddyl::ActionDataAbstract> runningModel_data = runningModel->createData();
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> runningDAM_data = runningDAM->createData();
  crocoddyl::DifferentialActionDataContactFwdDynamics* d =
      static_cast<crocoddyl::DifferentialActionDataContactFwdDynamics*>(runningDAM_data.get());
  boost::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data = actuation->createData();
  boost::shared_ptr<crocoddyl::CostDataAbstract> goalTrackingCost_data = goalTrackingCost->createData(&d->multibody);
  boost::shared_ptr<crocoddyl::CostDataAbstract> xRegCost_data = xRegCost->createData(&d->multibody);
  boost::shared_ptr<crocoddyl::CostDataAbstract> uRegCost_data = uRegCost->createData(&d->multibody);

  boost::shared_ptr<crocoddyl::CostDataSum> runningCostModel_data = runningCostModel->createData(&d->multibody);

  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activationQuad = xRegCost->get_activation();
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> activationQuad_data = activationQuad->createData();

  /********************************************************************/

  double duration = 0;
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) x1s;  // (T, state->rand());
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) x2s;  // (T, state->rand());
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) us;   // (T, state->rand());
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) dxs(T, Eigen::VectorXd::Zero(2 * model.nv));

  for (size_t i = 0; i < T; ++i) {
    x1s.push_back(state->rand());
    x2s.push_back(state->rand());
    us.push_back(Eigen::VectorXd(actuation->get_nu()));
  }

  /*********************State**********************************/

  duration = 0;
  timer.reset();
  SMOOTH(T) {
    pinocchio::difference(model, x1s[_smooth].head(model.nq), x2s[_smooth].head(model.nq),
                          dxs[_smooth].head(model.nv));
  }
  duration = timer.get_us_duration();
  std::cout << "pinocchio::difference (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) {
    pinocchio::integrate(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv), x2s[_smooth].head(model.nq));
  }
  duration = timer.get_us_duration();
  std::cout << "pinocchio::integrate (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->diff(x1s[_smooth], x2s[_smooth], dxs[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.diff (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->integrate(x1s[_smooth], dxs[_smooth], x2s[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.integrate (in us):\t\t" << duration / T << " us" << std::endl;

  Eigen::MatrixXd Jfirst(2 * model.nv, 2 * model.nv), Jsecond(2 * model.nv, 2 * model.nv);

  duration = 0;
  timer.reset();
  SMOOTH(T) {
    pinocchio::dIntegrate(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv),
                          Jsecond.bottomLeftCorner(model.nv, model.nv), pinocchio::ARG1);
  }
  duration = timer.get_us_duration();
  std::cout << "pinocchio::dIntegrate ARG1 (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) {
    pinocchio::dIntegrate(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv),
                          Jsecond.bottomLeftCorner(model.nv, model.nv), pinocchio::ARG0);
  }
  duration = timer.get_us_duration();
  std::cout << "pinocchio::dIntegrate ARG0 (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  Eigen::MatrixXd Jin(Eigen::MatrixXd::Random(model.nv, 2 * model.nv));
  SMOOTH(T) {
    pinocchio::dIntegrateTransport(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv), Jin,
                                   pinocchio::ARG0);
  }
  duration = timer.get_us_duration();
  std::cout << "pin::dIntegrateTransport with aliasing(in us):\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  Jin = Eigen::MatrixXd::Random(model.nv, 2 * model.nv);
  Eigen::MatrixXd Jout(Eigen::MatrixXd::Random(model.nv, 2 * model.nv));
  SMOOTH(T) {
    pinocchio::dIntegrateTransport(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv), Jin, Jout,
                                   pinocchio::ARG0);
  }
  duration = timer.get_us_duration();
  std::cout << "pin::dIntegrateTransport w/o aliasing(in us):\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::both); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.Jdiff both (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::first); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.Jdiff first (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::second); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.Jdiff second (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond, crocoddyl::both); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.Jintegrate both (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond, crocoddyl::first); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.Jintegrate first (in us):\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond, crocoddyl::second); }
  duration = timer.get_us_duration();
  std::cout << "StateMultibody.Jintegrate second (in us):\t" << duration / T << " us" << std::endl;

  /**************************************************************/

  duration = 0;
  timer.reset();
  SMOOTH(T) { activationQuad->calc(activationQuad_data, dxs[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ActivationModelQuad.calc (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { activationQuad->calcDiff(activationQuad_data, dxs[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ActivationModelQuad.calcdiff (in us):\t\t" << duration / T << " us" << std::endl;

  /*************************************Actuation*******************/

  duration = 0;
  timer.reset();
  SMOOTH(T) { actuation->calc(actuation_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ActuationModelFloatingBase.calc (in us):\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { actuation->calcDiff(actuation_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ActuationModelFloatingBase.calcdiff (in us):\t" << duration / T << " us" << std::endl;

  /*******************************Cost****************************/
  duration = 0;
  timer.reset();
  SMOOTH(T) { goalTrackingCost->calc(goalTrackingCost_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelFramePlacement.calc (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { goalTrackingCost->calcDiff(goalTrackingCost_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelFramePlacement.calcdiff (in us):\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { xRegCost->calc(xRegCost_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelState.calc (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { xRegCost->calcDiff(xRegCost_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelState.calcdiff (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { uRegCost->calc(uRegCost_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelControl.calc (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { uRegCost->calcDiff(uRegCost_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelControl.calcdiff (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { runningCostModel->calc(runningCostModel_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelSum calc (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { runningCostModel->calcDiff(runningCostModel_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "CostModelSum calcdiff (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { runningDAM->calc(runningDAM_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ContactDAM.calc (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { runningDAM->calcDiff(runningDAM_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "Contact DAM calcDiff (in us):\t\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { runningModel->calc(runningModel_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ContactDAM+EulerIAM calc (in us):\t\t" << duration / T << " us" << std::endl;

  duration = 0;
  timer.reset();
  SMOOTH(T) { runningModel->calcDiff(runningModel_data, x1s[_smooth], us[_smooth]); }
  duration = timer.get_us_duration();
  std::cout << "ContactDAM+EulerIAM calcDiff (in us):\t\t" << duration / T << " us" << std::endl;
}
