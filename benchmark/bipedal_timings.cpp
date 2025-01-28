///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, CTU, INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <example-robot-data/path.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

#define SMOOTH(s) for (size_t _smooth = 0; _smooth < s; ++_smooth)

#define STDDEV(vec) \
  std::sqrt(((vec - vec.mean())).square().sum() / ((double)vec.size() - 1))
#define AVG(vec) (vec.mean())

void printStatistics(std::string name, Eigen::ArrayXd duration) {
  std::cout << "  " << std::left << std::setw(42) << name << std::left
            << std::setw(15) << AVG(duration) << std::left << std::setw(15)
            << STDDEV(duration) << std::left << std::setw(15)
            << duration.maxCoeff() << std::left << std::setw(15)
            << duration.minCoeff() << std::endl;
}

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

  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR
                              "/talos_data/robots/talos_reduced.urdf",
                              pinocchio::JointModelFreeFlyer(), model);
  model.lowerPositionLimit.head<7>().array() = -1;
  model.upperPositionLimit.head<7>().array() = 1.;

  pinocchio::srdf::loadReferenceConfigurations(
      model, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf", false);
  const std::string RF = "leg_right_6_joint";
  const std::string LF = "leg_left_6_joint";

  /*************************PINOCCHIO MODEL**************/

  /************************* SETUP ***********************/
  crocoddyl::Timer timer;
  std::cout << "NQ: " << model.nq << std::endl;

  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::make_shared<crocoddyl::StateMultibody>(
          std::make_shared<pinocchio::Model>(model));
  std::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation =
      std::make_shared<crocoddyl::ActuationModelFloatingBase>(state);

  Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
  Eigen::VectorXd x0(state->get_nx());
  x0 << q0, Eigen::VectorXd::Random(state->get_nv());
  Eigen::MatrixXd Jfirst(2 * model.nv, 2 * model.nv),
      Jsecond(2 * model.nv, 2 * model.nv);

  std::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
      std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelFramePlacement>(
                     state, model.getFrameId("arm_right_7_joint"),
                     pinocchio::SE3(Eigen::Matrix3d::Identity(),
                                    Eigen::Vector3d(.0, .0, .4)),
                     actuation->get_nu()));

  std::shared_ptr<crocoddyl::CostModelAbstract> xRegCost =
      std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelState>(
                     state, actuation->get_nu()));
  std::shared_ptr<crocoddyl::CostModelAbstract> uRegCost =
      std::make_shared<crocoddyl::CostModelResidual>(
          state, std::make_shared<crocoddyl::ResidualModelControl>(
                     state, actuation->get_nu()));

  std::shared_ptr<crocoddyl::CostModelSum> runningCostModel =
      std::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  std::shared_ptr<crocoddyl::CostModelSum> terminalCostModel =
      std::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  runningCostModel->addCost("gripperPose", goalTrackingCost, 1);
  runningCostModel->addCost("xReg", xRegCost, 1e-4);
  runningCostModel->addCost("uReg", uRegCost, 1e-4);
  terminalCostModel->addCost("gripperPose", goalTrackingCost, 1);

  std::shared_ptr<crocoddyl::ContactModelMultiple> contact_models =
      std::make_shared<crocoddyl::ContactModelMultiple>(state,
                                                        actuation->get_nu());

  std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model6D =
      std::make_shared<crocoddyl::ContactModel6D>(
          state, model.getFrameId(RF), pinocchio::SE3::Identity(),
          pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
          Eigen::Vector2d(0., 50.));
  contact_models->addContact(
      model.frames[model.getFrameId(RF)].name + "_contact",
      support_contact_model6D);

  std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model3D =
      std::make_shared<crocoddyl::ContactModel3D>(
          state, model.getFrameId(LF), Eigen::Vector3d::Zero(),
          pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
          Eigen::Vector2d(0., 50.));
  contact_models->addContact(
      model.frames[model.getFrameId(LF)].name + "_contact",
      support_contact_model3D);

  std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
      runningDAM = std::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state, actuation, contact_models, runningCostModel);

  std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
      terminalDAM = std::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state, actuation, contact_models, terminalCostModel);

  std::shared_ptr<crocoddyl::ActionModelAbstract> runningModelWithEuler =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(runningDAM, 1e-3);
  std::shared_ptr<crocoddyl::ActionModelAbstract> runningModelWithRK4 =
      std::make_shared<crocoddyl::IntegratedActionModelRK>(
          runningDAM, crocoddyl::RKType::four, 1e-3);
  std::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminalDAM,
                                                              1e-3);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >
      runningModelsWithEuler(N, runningModelWithEuler);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >
      runningModelsWithRK4(N, runningModelWithRK4);

  std::shared_ptr<crocoddyl::ShootingProblem> problemWithEuler =
      std::make_shared<crocoddyl::ShootingProblem>(x0, runningModelsWithEuler,
                                                   terminalModel);
  std::shared_ptr<crocoddyl::ShootingProblem> problemWithRK4 =
      std::make_shared<crocoddyl::ShootingProblem>(x0, runningModelsWithRK4,
                                                   terminalModel);
  std::vector<Eigen::VectorXd> xs(N + 1, x0);

  /***************************************************************/

  std::shared_ptr<crocoddyl::ActionDataAbstract> runningModelWithEuler_data =
      runningModelWithEuler->createData();
  std::shared_ptr<crocoddyl::ActionDataAbstract> runningModelWithRK4_data =
      runningModelWithRK4->createData();
  std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> runningDAM_data =
      runningDAM->createData();
  crocoddyl::DifferentialActionDataContactFwdDynamics* d =
      static_cast<crocoddyl::DifferentialActionDataContactFwdDynamics*>(
          runningDAM_data.get());
  std::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data =
      actuation->createData();
  std::shared_ptr<crocoddyl::CostDataAbstract> goalTrackingCost_data =
      goalTrackingCost->createData(&d->multibody);
  std::shared_ptr<crocoddyl::CostDataAbstract> xRegCost_data =
      xRegCost->createData(&d->multibody);
  std::shared_ptr<crocoddyl::CostDataAbstract> uRegCost_data =
      uRegCost->createData(&d->multibody);

  std::shared_ptr<crocoddyl::CostDataSum> runningCostModel_data =
      runningCostModel->createData(&d->multibody);

  std::shared_ptr<crocoddyl::ActivationModelAbstract> activationQuad =
      xRegCost->get_activation();
  std::shared_ptr<crocoddyl::ActivationDataAbstract> activationQuad_data =
      activationQuad->createData();

  /********************************************************************/

  Eigen::ArrayXd duration(T);

  std::vector<Eigen::VectorXd> x0s;
  std::vector<Eigen::VectorXd> u0s;
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) x1s;  // (T, state->rand());
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) x2s;  // (T, state->rand());
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd) us;   // (T, state->rand());
  PINOCCHIO_ALIGNED_STD_VECTOR(Eigen::VectorXd)
  dxs(T, Eigen::VectorXd::Zero(2 * model.nv));

  for (size_t i = 0; i < T; ++i) {
    x1s.push_back(state->rand());
    x2s.push_back(state->rand());
    us.push_back(Eigen::VectorXd(actuation->get_nu()));
  }
  for (size_t i = 0; i < N; ++i) {
    x0s.push_back(state->rand());
    u0s.push_back(Eigen::VectorXd(actuation->get_nu()));
  }
  x0s.push_back(state->rand());

  /*********************State**********************************/
  std::cout << std::left << std::setw(42) << "Function call"
            << "  " << std::left << std::setw(15) << "AVG (us)" << std::left
            << std::setw(15) << "STDDEV (us)" << std::left << std::setw(15)
            << "MAX (us)" << std::left << std::setw(15) << "MIN (us)"
            << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::difference(model, x1s[_smooth].head(model.nq),
                          x2s[_smooth].head(model.nq),
                          dxs[_smooth].head(model.nv));
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pinocchio" << std::endl;
  printStatistics("difference", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::integrate(model, x1s[_smooth].head(model.nq),
                         dxs[_smooth].head(model.nv),
                         x2s[_smooth].head(model.nq));
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("integrate", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrate(
        model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv),
        Jsecond.bottomLeftCorner(model.nv, model.nv), pinocchio::ARG1);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("dIntegrate ARG1", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrate(
        model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv),
        Jsecond.bottomLeftCorner(model.nv, model.nv), pinocchio::ARG0);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("dIntegrate ARG0", duration);

  duration.setZero();
  Eigen::MatrixXd Jin(Eigen::MatrixXd::Random(model.nv, 2 * model.nv));
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrateTransport(model, x1s[_smooth].head(model.nq),
                                   dxs[_smooth].head(model.nv), Jin,
                                   pinocchio::ARG0);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("dIntegrateTransport with aliasing", duration);

  duration.setZero();
  Jin = Eigen::MatrixXd::Random(model.nv, 2 * model.nv);
  Eigen::MatrixXd Jout(Eigen::MatrixXd::Random(model.nv, 2 * model.nv));
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrateTransport(model, x1s[_smooth].head(model.nq),
                                   dxs[_smooth].head(model.nv), Jin, Jout,
                                   pinocchio::ARG0);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("dIntegrateTransport w/o aliasing", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->diff(x1s[_smooth], x2s[_smooth], dxs[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "StateMultibody" << std::endl;
  printStatistics("diff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->integrate(x1s[_smooth], dxs[_smooth], x2s[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("integrate", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::both);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("Jdiff both", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::first);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("Jdiff first", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond,
                 crocoddyl::second);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("Jdiff second", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond,
                      crocoddyl::both);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("Jintegrate both", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond,
                      crocoddyl::first);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("Jintegrate first", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond,
                      crocoddyl::second);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("Jintegrate second", duration);

  /**************************************************************/

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    activationQuad->calc(activationQuad_data, dxs[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "ActivationModelQuad" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    activationQuad->calcDiff(activationQuad_data, dxs[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  /*************************************Actuation*******************/

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    actuation->calc(actuation_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "ActuationModelFloatingBase" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    actuation->calcDiff(actuation_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  /*******************************Cost****************************/
  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    goalTrackingCost->calc(goalTrackingCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "CostModelFramePlacement" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    goalTrackingCost->calcDiff(goalTrackingCost_data, x1s[_smooth],
                               us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    xRegCost->calc(xRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "CostModelState" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    xRegCost->calcDiff(xRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    uRegCost->calc(uRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "CostModelControl" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    uRegCost->calcDiff(uRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningCostModel->calc(runningCostModel_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "CostModelSum" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningCostModel->calcDiff(runningCostModel_data, x1s[_smooth],
                               us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningDAM->calc(runningDAM_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "ContactFwdDynamics" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningDAM->calcDiff(runningDAM_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningModelWithEuler->calc(runningModelWithEuler_data, x1s[_smooth],
                                us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "ContactFwdDynamics+Euler" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningModelWithEuler->calcDiff(runningModelWithEuler_data, x1s[_smooth],
                                    us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningModelWithRK4->calc(runningModelWithRK4_data, x1s[_smooth],
                              us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "ContactFwdDynamics+RK4" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningModelWithRK4->calcDiff(runningModelWithRK4_data, x1s[_smooth],
                                  us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration = Eigen::ArrayXd(T / N);
  SMOOTH(T / N) {
    timer.reset();
    problemWithEuler->calc(x0s, u0s);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "Problem+Euler" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T / N) {
    timer.reset();
    problemWithEuler->calcDiff(x0s, u0s);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);

  duration.setZero();
  SMOOTH(T / N) {
    timer.reset();
    problemWithRK4->calc(x0s, u0s);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "Problem+RK4" << std::endl;
  printStatistics("calc", duration);

  duration.setZero();
  SMOOTH(T / N) {
    timer.reset();
    problemWithRK4->calcDiff(x0s, u0s);
    duration[_smooth] = timer.get_us_duration();
  }
  printStatistics("calcDiff", duration);
}
