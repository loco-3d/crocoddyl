///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifdef WITH_MULTITHREADING
#include <omp.h>
#define NTHREAD 4
#else
#define NTHREAD 1
#endif

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
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

#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / (vec.size() - 1)) * 1000
#define AVG(vec) (vec.mean()) * 1000.


int main(int argc, char* argv[]) {
  bool CALLBACKS = false;
  unsigned int N = 100;  // number of nodes
  unsigned int T = 1e2;  // number of trials
  unsigned int MAXITER = 1;
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
  std::cout << "Number of nodes: " << N << std::endl;

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

  crocoddyl::SolverDDP ddp(problem);
  ddp.setCandidate(xs, us, false);

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
  typedef crocoddyl::MathBaseTpl<ADScalar>::MatrixXs ADMatrixXs;
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
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > cg_runningData = cg_runningModel->createData();
  boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > runningData = runningModel->createData();
  VectorXs x_rand = cg_runningModel->get_state()->rand();
  VectorXs u_rand = VectorXs::Random(cg_runningModel->get_nu());


  runningModel->calc(runningData, x_rand, u_rand);
  runningModel->calcDiff(runningData, x_rand, u_rand);
  cg_runningModel->calc(cg_runningData, x_rand, u_rand);

  cg_runningModel->calcDiff(cg_runningData, x_rand, u_rand);

  Eigen::ArrayXd duration_cd(T);
  Eigen::ArrayXd avg_cd(NTHREAD);
  Eigen::ArrayXd stddev_cd(NTHREAD);

  Eigen::ArrayXd duration_cd_wo_calc(T);
  Eigen::ArrayXd avg_cd_wo_calc(NTHREAD);
  Eigen::ArrayXd stddev_cd_wo_calc(NTHREAD);

  Eigen::ArrayXd duration_calc(T);
  Eigen::ArrayXd avg_calc(NTHREAD);
  Eigen::ArrayXd stddev_calc(NTHREAD);

  Eigen::ArrayXd duration_dcalc(T);
  Eigen::ArrayXd avg_dcalc(NTHREAD);
  Eigen::ArrayXd stddev_dcalc(NTHREAD);

  Eigen::ArrayXd duration_dcalcpin(T);
  Eigen::ArrayXd avg_dcalcpin(NTHREAD);
  Eigen::ArrayXd stddev_dcalcpin(NTHREAD);

  Eigen::ArrayXd duration_diffcalcpin(T);
  Eigen::ArrayXd avg_diffcalcpin(NTHREAD);
  Eigen::ArrayXd stddev_diffcalcpin(NTHREAD);

  problem->calc(xs, us);

  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    duration_cd.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    // Timings pyrene-arm-calc+calcDiff
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      // start of calcDiff function
      for (unsigned int j = 0; j < N; ++j) {
        runningModels[j]->calcDiff(problem->get_runningDatas()[j], xs[j], us[j]);
      }
      // terminalModel->calcDiff(problem->terminal_data_, xs.back());
      // end of calcdiff function
      duration_cd[i] = timer.get_duration();
    }
    avg_cd[ithread] = AVG(duration_cd);
    stddev_cd[ithread] = STDDEV(duration_cd);
    std::cout << ithread + 1 << " threaded calcDiff [mean +- stddev in us]: " << avg_cd[ithread] << " +- "
              << stddev_cd[ithread] << " (per nodes/thread: " << avg_cd[ithread] * (ithread + 1) / N << " +- "
              << stddev_cd[ithread] * (ithread + 1) / N << ")" << std::endl;
  }

  // CALC Timings
  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    duration_calc.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    // Timings pyrene-biped-calc+calcDiff
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      // start of calcDiff function
      for (unsigned int j = 0; j < N; ++j) {
        runningModels[j]->calc(problem->get_runningDatas()[j], xs[j], us[j]);
      }
      // end of calcdiff function
      duration_calc[i] = timer.get_duration();
    }
    avg_calc[ithread] = AVG(duration_calc);
    stddev_calc[ithread] = STDDEV(duration_calc);
    std::cout << ithread + 1 << " threaded calc [mean +- stddev in us]: " << avg_calc[ithread] << " +- "
              << stddev_calc[ithread] << " (per nodes/thread: " << avg_calc[ithread] * (ithread + 1) / N << " +- "
              << stddev_calc[ithread] * (ithread + 1) / N << ")" << std::endl;
  }

  // differential CALC Timings
  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    duration_dcalc.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    // Timings pyrene-biped-calc+calcDiff
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;

#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      // start of calcDiff function
      for (unsigned int j = 0; j < N; ++j) {
        boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> m =
            boost::static_pointer_cast<crocoddyl::IntegratedActionModelEuler>(problem->get_runningModels()[j]);
        boost::shared_ptr<crocoddyl::IntegratedActionDataEuler> d =
            boost::static_pointer_cast<crocoddyl::IntegratedActionDataEuler>(problem->get_runningDatas()[j]);

        m->get_differential()->calc(d->differential, xs[j], us[j]);
      }
      // end of calcdiff function
      duration_dcalc[i] = timer.get_duration();
    }
    avg_dcalc[ithread] = AVG(duration_dcalc);
    stddev_dcalc[ithread] = STDDEV(duration_dcalc);
    std::cout << ithread + 1 << " threaded differential calc [mean +- stddev in us]: " << avg_dcalc[ithread] << " +- "
              << stddev_dcalc[ithread] << " (per nodes/thread: " << avg_dcalc[ithread] * (ithread + 1) / N << " +- "
              << stddev_dcalc[ithread] * (ithread + 1) / N << ")" << std::endl;
  }

  ddp.calcDiff();
  Eigen::ArrayXd duration_bp(T);
  // std::cout << "Starting timing backwardpass"<<std::endl;

  // Timings pyrene-biped-backwardPass
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    ddp.backwardPass();
    duration_bp[i] = timer.get_duration();
  }
  double avg_bp = AVG(duration_bp);
  double stddev_bp = STDDEV(duration_bp);
  std::cout << "backwardPass [mean +- stddev in us]: " << avg_bp << " +- " << stddev_bp
            << " (per nodes: " << avg_bp / N << " +- " << stddev_bp / N << ")" << std::endl;

  Eigen::ArrayXd duration_fp(T);
  // std::cout << "Starting likwid forward pass"<<std::endl;

  // std::cout << "Starting timing forwardpass"<<std::endl;

  // Timings pyrene-biped-forwardPass
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    ddp.forwardPass(0.5);
    duration_fp[i] = timer.get_duration();
  }
  double avg_fp = AVG(duration_fp);
  double stddev_fp = STDDEV(duration_fp);
  std::cout << "forwardPass [mean +- stddev in us]: " << avg_fp << " +- " << stddev_fp << " (per nodes: " << avg_fp / N
            << " +- " << stddev_fp / N << ")" << std::endl;

  /**********************************************/
  /***************************CHECKING CODE GEN TIMINGS*********************/

  Eigen::ArrayXd cg_duration_cd(T);
  Eigen::ArrayXd cg_avg_cd(NTHREAD);
  Eigen::ArrayXd cg_stddev_cd(NTHREAD);

  Eigen::ArrayXd cg_duration_cd_wo_calc(T);
  Eigen::ArrayXd cg_avg_cd_wo_calc(NTHREAD);
  Eigen::ArrayXd cg_stddev_cd_wo_calc(NTHREAD);

  Eigen::ArrayXd cg_duration_calc(T);
  Eigen::ArrayXd cg_avg_calc(NTHREAD);
  Eigen::ArrayXd cg_stddev_calc(NTHREAD);

  Eigen::ArrayXd cg_duration_dcalc(T);
  Eigen::ArrayXd cg_avg_dcalc(NTHREAD);
  Eigen::ArrayXd cg_stddev_dcalc(NTHREAD);

  Eigen::ArrayXd cg_duration_dcalcpin(T);
  Eigen::ArrayXd cg_avg_dcalcpin(NTHREAD);
  Eigen::ArrayXd cg_stddev_dcalcpin(NTHREAD);

  Eigen::ArrayXd cg_duration_diffcalcpin(T);
  Eigen::ArrayXd cg_avg_diffcalcpin(NTHREAD);
  Eigen::ArrayXd cg_stddev_diffcalcpin(NTHREAD);

  cg_problem->calc(xs, us);

  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    cg_duration_cd.setZero();
#ifdef WITH_MULTITHREADING
    omp_set_num_threads(ithread + 1);
#endif
    // Timings pyrene-biped-calc+calcDiff
    for (unsigned int i = 0; i < T; ++i) {
      crocoddyl::Timer timer;

#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
      // start of calcDiff function
      for (unsigned int j = 0; j < N; ++j) {
        cg_runningModels[j]->calcDiff(cg_problem->get_runningDatas()[j], xs[j], us[j]);
      }
      // terminalModel->calcDiff(problem->terminal_data_, xs.back());
      // end of calcdiff function
      cg_duration_cd[i] = timer.get_duration();
    }
    cg_avg_cd[ithread] = AVG(cg_duration_cd);
    cg_stddev_cd[ithread] = STDDEV(cg_duration_cd);
    std::cout << ithread + 1 << " threaded calcDiff [mean +- stddev in us]: " << cg_avg_cd[ithread] << " +- "
              << cg_stddev_cd[ithread] << " (per nodes/thread: " << cg_avg_cd[ithread] * (ithread + 1) / N << " +- "
              << cg_stddev_cd[ithread] * (ithread + 1) / N << ")" << std::endl;
  }

  // CALC Timings
  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    cg_duration_calc.setZero();
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
      // end of calcdiff function
      cg_duration_calc[i] = timer.get_duration();
    }
    cg_avg_calc[ithread] = AVG(cg_duration_calc);
    cg_stddev_calc[ithread] = STDDEV(cg_duration_calc);
    std::cout << ithread + 1 << " threaded calc [mean +- stddev in us]: " << cg_avg_calc[ithread] << " +- "
              << cg_stddev_calc[ithread] << " (per nodes/thread: " << cg_avg_calc[ithread] * (ithread + 1) / N
              << " +- " << cg_stddev_calc[ithread] * (ithread + 1) / N << ")" << std::endl;
  }
  
}
