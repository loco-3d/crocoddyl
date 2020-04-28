///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef WITH_MULTITHREADING
#include <omp.h>
#define NUM_THREADS WITH_NTHREADS
#else
#define NUM_THREADS 1
#endif

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

#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / (static_cast<double>(vec.size()) - 1))
#define AVG(vec) (vec.mean())

int main(int argc, char* argv[]) {
  unsigned int N = 100;  // number of nodes
  unsigned int T = 1e3;  // number of trials
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  /**************************DOUBLE**********************/
  /**************************DOUBLE**********************/

  pinocchio::Model model;

  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf",
                              pinocchio::JointModelFreeFlyer(), model);
  model.lowerPositionLimit.head<7>().array() = -1;
  model.upperPositionLimit.head<7>().array() = 1.;

  std::cout << "NQ: " << model.nq << std::endl;
  std::cout << "Number of nodes: " << N << std::endl << std::endl;

  pinocchio::srdf::loadReferenceConfigurations(model, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                               false);
  const std::string RF = "leg_right_6_joint";
  const std::string LF = "leg_left_6_joint";

  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));

  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation =
      boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);

  Eigen::VectorXd x0(state->rand());

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

  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));

  for (unsigned int i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem->get_runningDatas()[i];
    model->quasiStatic(data, us[i], x0);
  }

  /**************************************Start ADScalar*******************************/
  /**************************************Start ADScalar*******************************/
  /**************************************Start ADScalar*******************************/
  /**************************************Start ADScalar*******************************/
  /**************************************Start ADScalar*******************************/
  /**************************************Start ADScalar*******************************/

  typedef double Scalar;
  typedef crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef crocoddyl::MathBaseTpl<ADScalar>::VectorXs ADVectorXs;
  typedef crocoddyl::MathBaseTpl<ADScalar>::Vector2s ADVector2s;
  typedef crocoddyl::MathBaseTpl<ADScalar>::Vector3s ADVector3s;
  typedef crocoddyl::MathBaseTpl<ADScalar>::Matrix3s ADMatrix3s;
  typedef crocoddyl::FramePlacementTpl<ADScalar> ADFramePlacement;
  typedef crocoddyl::FrameTranslationTpl<ADScalar> ADFrameTranslation;
  typedef crocoddyl::CostModelAbstractTpl<ADScalar> ADCostModelAbstract;
  typedef crocoddyl::CostModelFramePlacementTpl<ADScalar> ADCostModelFramePlacement;
  typedef crocoddyl::CostModelStateTpl<ADScalar> ADCostModelState;
  typedef crocoddyl::CostModelControlTpl<ADScalar> ADCostModelControl;
  typedef crocoddyl::CostModelSumTpl<ADScalar> ADCostModelSum;
  typedef crocoddyl::ContactModelAbstractTpl<ADScalar> ADContactModelAbstract;
  typedef crocoddyl::ContactModelMultipleTpl<ADScalar> ADContactModelMultiple;
  typedef crocoddyl::ContactModel3DTpl<ADScalar> ADContactModel3D;
  typedef crocoddyl::ContactModel6DTpl<ADScalar> ADContactModel6D;
  typedef crocoddyl::ActionModelAbstractTpl<ADScalar> ADActionModelAbstract;
  typedef crocoddyl::ActuationModelFloatingBaseTpl<ADScalar> ADActuationModelFloatingBase;
  typedef crocoddyl::DifferentialActionModelContactFwdDynamicsTpl<ADScalar>
      ADDifferentialActionModelContactFwdDynamics;
  typedef crocoddyl::IntegratedActionModelEulerTpl<ADScalar> ADIntegratedActionModelEuler;

  pinocchio::ModelTpl<ADScalar> ad_model(model.cast<ADScalar>());
  boost::shared_ptr<crocoddyl::StateMultibodyTpl<ADScalar> > ad_state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<ADScalar> >(
          boost::make_shared<pinocchio::ModelTpl<ADScalar> >(ad_model));

  ADVectorXs ad_x0(x0.cast<ADScalar>());

  boost::shared_ptr<ADActuationModelFloatingBase> ad_actuation =
      boost::make_shared<ADActuationModelFloatingBase>(ad_state);

  ADFramePlacement ad_Mref(
      ad_model.getFrameId("arm_right_7_joint"),
      pinocchio::SE3Tpl<ADScalar>(ADMatrix3s::Identity(), ADVector3s(ADScalar(.0), ADScalar(.0), ADScalar(.4))));

  boost::shared_ptr<ADCostModelAbstract> ad_goalTrackingCost =
      boost::make_shared<ADCostModelFramePlacement>(ad_state, ad_Mref, ad_actuation->get_nu());

  boost::shared_ptr<ADCostModelAbstract> ad_xRegCost =
      boost::make_shared<ADCostModelState>(ad_state, ad_actuation->get_nu());
  boost::shared_ptr<ADCostModelAbstract> ad_uRegCost =
      boost::make_shared<ADCostModelControl>(ad_state, ad_actuation->get_nu());

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<ADCostModelSum> ad_runningCostModel =
      boost::make_shared<ADCostModelSum>(ad_state, ad_actuation->get_nu());
  boost::shared_ptr<ADCostModelSum> ad_terminalCostModel =
      boost::make_shared<ADCostModelSum>(ad_state, ad_actuation->get_nu());

  // Then let's added the running and terminal cost functions
  ad_runningCostModel->addCost("gripperPose", ad_goalTrackingCost, ADScalar(1));
  ad_runningCostModel->addCost("xReg", ad_xRegCost, ADScalar(1e-4));
  ad_runningCostModel->addCost("uReg", ad_uRegCost, ADScalar(1e-4));
  ad_terminalCostModel->addCost("gripperPose", ad_goalTrackingCost, ADScalar(1));

  boost::shared_ptr<ADContactModelMultiple> ad_contact_models =
      boost::make_shared<ADContactModelMultiple>(ad_state, ad_actuation->get_nu());

  ADFramePlacement ad_xref(ad_model.getFrameId(RF), pinocchio::SE3Tpl<ADScalar>::Identity());
  boost::shared_ptr<ADContactModelAbstract> ad_support_contact_model6D = boost::make_shared<ADContactModel6D>(
      ad_state, ad_xref, ad_actuation->get_nu(), ADVector2s(ADScalar(0.), ADScalar(50.)));
  ad_contact_models->addContact(ad_model.frames[ad_model.getFrameId(RF)].name + "_contact",
                                ad_support_contact_model6D);

  ADFrameTranslation ad_x2ref(ad_model.getFrameId(LF), ADVector3s::Zero());
  boost::shared_ptr<ADContactModelAbstract> ad_support_contact_model3D = boost::make_shared<ADContactModel3D>(
      ad_state, ad_x2ref, ad_actuation->get_nu(), ADVector2s(ADScalar(0.), ADScalar(50.)));
  ad_contact_models->addContact(ad_model.frames[ad_model.getFrameId(LF)].name + "_contact",
                                ad_support_contact_model3D);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<ADDifferentialActionModelContactFwdDynamics> ad_runningDAM =
      boost::make_shared<ADDifferentialActionModelContactFwdDynamics>(ad_state, ad_actuation, ad_contact_models,
                                                                      ad_runningCostModel);

  boost::shared_ptr<ADDifferentialActionModelContactFwdDynamics> ad_terminalDAM =
      boost::make_shared<ADDifferentialActionModelContactFwdDynamics>(ad_state, ad_actuation, ad_contact_models,
                                                                      ad_terminalCostModel);

  // ADVectorXs ad_armature(ad_state->get_nq());
  // ad_armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  // ad_runningDAM->set_armature(ad_armature);
  // ad_terminalDAM->set_armature(ad_armature);
  boost::shared_ptr<ADActionModelAbstract> ad_runningModel =
      boost::make_shared<ADIntegratedActionModelEuler>(ad_runningDAM, ADScalar(1e-3));
  boost::shared_ptr<ADActionModelAbstract> ad_terminalModel =
      boost::make_shared<ADIntegratedActionModelEuler>(ad_terminalDAM, ADScalar(1e-3));

  /****************************/

  // For calculation and for the ShootingProblem!!

  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > cg_runningModel =
      boost::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar> >(ad_runningModel, runningModel, "biped_running");
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > cg_terminalModel =
      boost::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar> >(ad_terminalModel, terminalModel, "biped_terminal");
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > > cg_runningModels(N, cg_runningModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> cg_problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, cg_runningModels, cg_terminalModel);
  for (unsigned int i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& cg_model = cg_problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> >& cg_data = cg_problem->get_runningDatas()[i];
    cg_model->quasiStatic(cg_data, us[i], x0);
  }

  // Check that code-generated action model is the same as original.
  /**************************************************************************/
  crocoddyl::SolverDDP ddp(problem);
  crocoddyl::SolverDDP cg_ddp(cg_problem);
  ddp.setCandidate(xs, us, false);
  cg_ddp.setCandidate(xs, us, false);
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > cg_runningData = cg_runningModel->createData();
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > runningData = runningModel->createData();
  VectorXs x_rand = cg_runningModel->get_state()->rand();
  VectorXs u_rand = VectorXs::Random(cg_runningModel->get_nu());
  runningModel->calc(runningData, x_rand, u_rand);
  runningModel->calcDiff(runningData, x_rand, u_rand);
  cg_runningModel->calc(cg_runningData, x_rand, u_rand);
  cg_runningModel->calcDiff(cg_runningData, x_rand, u_rand);
  assert_pretty(cg_runningData->xnext.isApprox(runningData->xnext), "Problem in xnext");
  assert_pretty(cg_runningData->cost == runningData->cost, "Problem in cost");
  assert_pretty(cg_runningData->Lx.isApprox(runningData->Lx), "Problem in Lx");
  assert_pretty(cg_runningData->Lu.isApprox(runningData->Lu), "Problem in Lu");
  assert_pretty(cg_runningData->Lxx.isApprox(runningData->Lxx), "Problem in Lxx");
  assert_pretty(cg_runningData->Lxu.isApprox(runningData->Lxu), "Problem in Lxu");
  assert_pretty(cg_runningData->Luu.isApprox(runningData->Luu), "Problem in Luu");
  assert_pretty(cg_runningData->Fx.isApprox(runningData->Fx), "Problem in Fx");
  assert_pretty(cg_runningData->Fu.isApprox(runningData->Fu), "Problem in Fu");

  /*******************************************************************************/
  /*********************************** TIMINGS ***********************************/
  Eigen::ArrayXd duration(T);
  Eigen::ArrayXd avg(NUM_THREADS);
  Eigen::ArrayXd stddev(NUM_THREADS);

  /*******************************************************************************/
  /****************************** ACTION MODEL TIMINGS ***************************/
  std::cout << "Without Code Generation:" << std::endl;
  problem->calc(xs, us);
  // calcDiff timings
  for (int ithread = 0; ithread < NUM_THREADS; ++ithread) {
    duration.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      for (unsigned int j = 0; j < N; ++j) {
        runningModels[j]->calcDiff(problem->get_runningDatas()[j], xs[j], us[j]);
      }
      duration[i] = timer.get_duration();
    }
    avg[ithread] = AVG(duration);
    stddev[ithread] = STDDEV(duration);
    std::cout << ithread + 1 << " threaded calcDiff [ms]:\t" << avg[ithread] << " +- " << stddev[ithread]
              << " (per nodes: " << avg[ithread] * (ithread + 1) / N << " +- " << stddev[ithread] * (ithread + 1) / N
              << ")" << std::endl;
  }

  // calc timings
  for (int ithread = 0; ithread < NUM_THREADS; ++ithread) {
    duration.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      for (unsigned int j = 0; j < N; ++j) {
        runningModels[j]->calc(problem->get_runningDatas()[j], xs[j], us[j]);
      }
      duration[i] = timer.get_duration();
    }
    avg[ithread] = AVG(duration);
    stddev[ithread] = STDDEV(duration);
    std::cout << ithread + 1 << " threaded calc [ms]:    \t" << avg[ithread] << " +- " << stddev[ithread]
              << " (per nodes: " << avg[ithread] * (ithread + 1) / N << " +- " << stddev[ithread] * (ithread + 1) / N
              << ")" << std::endl;
  }

  /*******************************************************************************/
  /************************* DIFFERENTIAL ACTION TIMINGS *************************/
  for (int ithread = 0; ithread < NUM_THREADS; ++ithread) {
    duration.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      for (unsigned int j = 0; j < N; ++j) {
        boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> m =
            boost::static_pointer_cast<crocoddyl::IntegratedActionModelEuler>(problem->get_runningModels()[j]);
        boost::shared_ptr<crocoddyl::IntegratedActionDataEuler> d =
            boost::static_pointer_cast<crocoddyl::IntegratedActionDataEuler>(problem->get_runningDatas()[j]);

        m->get_differential()->calc(d->differential, xs[j], us[j]);
      }
      duration[i] = timer.get_duration();
    }
    avg[ithread] = AVG(duration);
    stddev[ithread] = STDDEV(duration);
    std::cout << ithread + 1 << " threaded diff calc [ms]: \t" << avg[ithread] << " +- " << stddev[ithread]
              << " (per nodes: " << avg[ithread] * (ithread + 1) / N << " +- " << stddev[ithread] * (ithread + 1) / N
              << ")" << std::endl;
  }

  /*******************************************************************************/
  /******************* DDP BACKWARD AND FORWARD PASSES TIMINGS *******************/
  // Backward pass timings
  ddp.calcDiff();
  duration.setZero();
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    ddp.backwardPass();
    duration[i] = timer.get_duration();
  }
  double avg_bp = AVG(duration);
  double stddev_bp = STDDEV(duration);
  std::cout << "backwardPass [ms]:\t\t" << avg_bp << " +- " << stddev_bp << " (per nodes: " << avg_bp / N << " +- "
            << stddev_bp / N << ")" << std::endl;

  // Forward pass timings
  duration.setZero();
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    ddp.forwardPass(0.5);
    duration[i] = timer.get_duration();
  }
  double avg_fp = AVG(duration);
  double stddev_fp = STDDEV(duration);
  std::cout << "forwardPass [ms]: \t\t" << avg_fp << " +- " << stddev_fp << " (per nodes: " << avg_fp / N << " +- "
            << stddev_fp / N << ")" << std::endl;

  /*******************************************************************************/
  /*************************** CODE GENERATION TIMINGS ***************************/
  /*******************************************************************************/

  /*******************************************************************************/
  /****************************** ACTION MODEL TIMINGS ***************************/
  std::cout << std::endl << "With Code Generation:" << std::endl;
  // calcDiff timings
  cg_problem->calc(xs, us);
  for (int ithread = 0; ithread < NUM_THREADS; ++ithread) {
    duration.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      for (unsigned int j = 0; j < N; ++j) {
        cg_runningModels[j]->calcDiff(cg_problem->get_runningDatas()[j], xs[j], us[j]);
      }
      duration[i] = timer.get_duration();
    }
    avg[ithread] = AVG(duration);
    stddev[ithread] = STDDEV(duration);
    std::cout << ithread + 1 << " threaded calcDiff [ms]:\t" << avg[ithread] << " +- " << stddev[ithread]
              << " (per nodes: " << avg[ithread] * (ithread + 1) / N << " +- " << stddev[ithread] * (ithread + 1) / N
              << ")" << std::endl;
  }

  // calc timings
  for (int ithread = 0; ithread < NUM_THREADS; ++ithread) {
    duration.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      for (unsigned int j = 0; j < N; ++j) {
        cg_runningModels[j]->calc(cg_problem->get_runningDatas()[j], xs[j], us[j]);
      }
      duration[i] = timer.get_duration();
    }
    avg[ithread] = AVG(duration);
    stddev[ithread] = STDDEV(duration);
    std::cout << ithread + 1 << " threaded calc [ms]:    \t" << avg[ithread] << " +- " << stddev[ithread]
              << " (per nodes: " << avg[ithread] * (ithread + 1) / N << " +- " << stddev[ithread] * (ithread + 1) / N
              << ")" << std::endl;
  }

  /*******************************************************************************/
  /******************* DDP BACKWARD AND FORWARD PASSES TIMINGS *******************/
  // Backward pass timings
  cg_ddp.calcDiff();
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    cg_ddp.backwardPass();
    duration[i] = timer.get_duration();
  }
  avg_bp = AVG(duration);
  stddev_bp = STDDEV(duration);
  std::cout << "backwardPass [ms]:\t\t" << avg_bp << " +- " << stddev_bp << " (per nodes: " << avg_bp / N << " +- "
            << stddev_bp / N << ")" << std::endl;

  // Forward pass timings
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    cg_ddp.forwardPass(0.5);
    duration[i] = timer.get_duration();
  }
  avg_fp = AVG(duration);
  stddev_fp = STDDEV(duration);
  std::cout << "forwardPass [ms]: \t\t" << avg_fp << " +- " << stddev_fp << " (per nodes: " << avg_fp / N << " +- "
            << stddev_fp / N << ")" << std::endl;
}
