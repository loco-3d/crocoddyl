///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>

#include <example-robot-data/path.hpp>
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"

#include "crocoddyl/core/mathbase.hpp"

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#define SMOOTH(s) for (size_t _smooth = 0; _smooth < s; ++_smooth)

#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / ((double)vec.size() - 1))
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
  pinocchio::Model model_full, model;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf", model_full);
  std::vector<pinocchio::JointIndex> locked_joints;
  locked_joints.reserve(3);

  locked_joints.push_back(5);
  locked_joints.push_back(6);
  locked_joints.push_back(7);

  pinocchio::buildReducedModel(model_full, locked_joints, Eigen::VectorXd::Zero(model_full.nq), model);

  /*************************PINOCCHIO MODEL**************/

  /************************* SETUP ***********************/
  crocoddyl::Timer timer;
  std::cout << "NQ: " << model.nq << std::endl;

  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));
  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
      boost::make_shared<crocoddyl::ActuationModelFull>(state);

  Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
  Eigen::VectorXd x0(state->get_nx());
  x0 << q0, Eigen::VectorXd::Random(state->get_nv());

  crocoddyl::FramePlacement Mref(model.getFrameId("gripper_left_joint"),
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

  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> runningDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, runningCostModel);

  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> terminalDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, terminalCostModel);

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
  crocoddyl::DifferentialActionDataFreeFwdDynamics* d =
      static_cast<crocoddyl::DifferentialActionDataFreeFwdDynamics*>(runningDAM_data.get());
  boost::shared_ptr<crocoddyl::ActuationDataAbstract> actuation_data = actuation->createData();
  boost::shared_ptr<crocoddyl::CostDataAbstract> goalTrackingCost_data = goalTrackingCost->createData(&d->multibody);
  boost::shared_ptr<crocoddyl::CostDataAbstract> xRegCost_data = xRegCost->createData(&d->multibody);
  boost::shared_ptr<crocoddyl::CostDataAbstract> uRegCost_data = uRegCost->createData(&d->multibody);

  boost::shared_ptr<crocoddyl::CostDataSum> runningCostModel_data = runningCostModel->createData(&d->multibody);

  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activationQuad = xRegCost->get_activation();
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> activationQuad_data = activationQuad->createData();

  /********************************************************************/

  Eigen::ArrayXd duration(T);

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
  std::cout << "Function call: \t\t\t\t"
            << "AVG(in us)\t"
            << "STDDEV(in us)\t"
            << "MAX(in us)\t"
            << "MIN(in us)" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::difference(model, x1s[_smooth].head(model.nq), x2s[_smooth].head(model.nq),
                          dxs[_smooth].head(model.nv));
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pinocchio::difference :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::integrate(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv), x2s[_smooth].head(model.nq));
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pinocchio::integrate :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->diff(x1s[_smooth], x2s[_smooth], dxs[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "StateMultibody.diff :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->integrate(x1s[_smooth], dxs[_smooth], x2s[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "StateMultibody.integrate :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  Eigen::MatrixXd Jfirst(2 * model.nv, 2 * model.nv), Jsecond(2 * model.nv, 2 * model.nv);

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrate(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv),
                          Jsecond.bottomLeftCorner(model.nv, model.nv), pinocchio::ARG1);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pinocchio::dIntegrate ARG1 :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrate(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv),
                          Jsecond.bottomLeftCorner(model.nv, model.nv), pinocchio::ARG0);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pinocchio::dIntegrate ARG0 :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  Eigen::MatrixXd Jin(Eigen::MatrixXd::Random(model.nv, 2 * model.nv));
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrateTransport(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv), Jin,
                                   pinocchio::ARG0);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pin::dIntegrateTransport with aliasing:\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  Jin = Eigen::MatrixXd::Random(model.nv, 2 * model.nv);
  Eigen::MatrixXd Jout(Eigen::MatrixXd::Random(model.nv, 2 * model.nv));
  SMOOTH(T) {
    timer.reset();
    pinocchio::dIntegrateTransport(model, x1s[_smooth].head(model.nq), dxs[_smooth].head(model.nv), Jin, Jout,
                                   pinocchio::ARG0);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "pin::dIntegrateTransport w/o aliasing:\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::both);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "StateMultibody.Jdiff both :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::first);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "StateMultibody.Jdiff first :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jdiff(x1s[_smooth], x2s[_smooth], Jfirst, Jsecond, crocoddyl::second);
    duration[_smooth] = timer.get_us_duration();
  }
  std::cout << "StateMultibody.Jdiff second :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond, crocoddyl::both);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "StateMultibody.Jintegrate both :\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond, crocoddyl::first);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "StateMultibody.Jintegrate first :\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    state->Jintegrate(x1s[_smooth], dxs[_smooth], Jfirst, Jsecond, crocoddyl::second);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "StateMultibody.Jintegrate second :\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  /**************************************************************/

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    activationQuad->calc(activationQuad_data, dxs[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ActivationModelQuad.calc :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    activationQuad->calcDiff(activationQuad_data, dxs[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ActivationModelQuad.calcdiff :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  /*************************************Actuation*******************/

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    actuation->calc(actuation_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ActuationModelFloatingBase.calc :\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    actuation->calcDiff(actuation_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ActuationModelFloatingBase.calcdiff :\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  /*******************************Cost****************************/
  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    goalTrackingCost->calc(goalTrackingCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelFramePlacement.calc :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    goalTrackingCost->calcDiff(goalTrackingCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelFramePlacement.calcdiff :\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    xRegCost->calc(xRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelState.calc :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    xRegCost->calcDiff(xRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelState.calcdiff :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    uRegCost->calc(uRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelControl.calc :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    uRegCost->calcDiff(uRegCost_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelControl.calcdiff :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningCostModel->calc(runningCostModel_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelSum calc :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningCostModel->calcDiff(runningCostModel_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "CostModelSum calcdiff :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningDAM->calc(runningDAM_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ContactDAM.calc :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningDAM->calcDiff(runningDAM_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "Contact DAM calcDiff :\t\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningModel->calc(runningModel_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ContactDAM+EulerIAM calc :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;

  duration.setZero();
  SMOOTH(T) {
    timer.reset();
    runningModel->calcDiff(runningModel_data, x1s[_smooth], us[_smooth]);
    duration[_smooth] = timer.get_us_duration();
  }

  std::cout << "ContactDAM+EulerIAM calcDiff :\t\t" << AVG(duration) << " us\t" << STDDEV(duration) << " us\t"
            << duration.maxCoeff() << " us\t" << duration.minCoeff() << " us" << std::endl;
}
