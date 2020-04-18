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
#include <pinocchio/algorithm/model.hpp>

#include <example-robot-data/path.hpp>
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/codegen/action-base.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
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

#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / (double(vec.size()) - 1.)) * 1000
#define AVG(vec) (vec.mean()) * 1000.

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
  std::vector<pinocchio::JointIndex> locked_joints{5, 6, 7};
  pinocchio::buildReducedModel(model_full, locked_joints, Eigen::VectorXd::Zero(model_full.nq), model);
  std::cout << "NQ: " << model.nq << std::endl;
  std::cout << "Number of nodes: " << N << std::endl;

  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));

  Eigen::VectorXd q0 = Eigen::VectorXd::Random(state->get_nq());
  Eigen::VectorXd x0(state->get_nx());
  x0 << q0, Eigen::VectorXd::Random(state->get_nv());

  // Note that we need to include a cost model (i.e. set of cost functions) in
  // order to fully define the action model for our optimal control problem.
  // For this particular example, we formulate three running-cost functions:
  // goal-tracking cost, state and control regularization; and one terminal-cost:
  // goal cost. First, let's create the common cost functions.
  crocoddyl::FramePlacement Mref(model.getFrameId("gripper_left_joint"),
                                 pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.0, .0, .4)));
  boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost =
      boost::make_shared<crocoddyl::CostModelFramePlacement>(state, Mref);
  boost::shared_ptr<crocoddyl::CostModelAbstract> xRegCost = boost::make_shared<crocoddyl::CostModelState>(state);
  boost::shared_ptr<crocoddyl::CostModelAbstract> uRegCost = boost::make_shared<crocoddyl::CostModelControl>(state);

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);
  boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, 1);
  runningCostModel->addCost("xReg", xRegCost, 1e-4);
  runningCostModel->addCost("uReg", uRegCost, 1e-4);
  terminalCostModel->addCost("gripperPose", goalTrackingCost, 1);

  // We define an actuation model
  boost::shared_ptr<crocoddyl::ActuationModelFull> actuation =
      boost::make_shared<crocoddyl::ActuationModelFull>(state);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> runningDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, runningCostModel);

  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> terminalDAM =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(state, actuation, terminalCostModel);

  // Eigen::VectorXd armature(state->get_nq());
  // armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.;
  // runningDAM->set_armature(armature);
  // terminalDAM->set_armature(armature);
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

  /**************************ADScalar**********************/
  /**************************ADScalar**********************/
  /**************************ADScalar**********************/
  typedef double Scalar;
  typedef crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef crocoddyl::MathBaseTpl<ADScalar>::VectorXs ADVectorXs;
  typedef crocoddyl::MathBaseTpl<ADScalar>::Vector3s ADVector3s;
  typedef crocoddyl::MathBaseTpl<ADScalar>::Matrix3s ADMatrix3s;
  typedef crocoddyl::FramePlacementTpl<ADScalar> ADFramePlacement;
  typedef crocoddyl::CostModelAbstractTpl<ADScalar> ADCostModelAbstract;
  typedef crocoddyl::CostModelFramePlacementTpl<ADScalar> ADCostModelFramePlacement;
  typedef crocoddyl::CostModelStateTpl<ADScalar> ADCostModelState;
  typedef crocoddyl::CostModelControlTpl<ADScalar> ADCostModelControl;
  typedef crocoddyl::CostModelSumTpl<ADScalar> ADCostModelSum;
  typedef crocoddyl::ActionModelAbstractTpl<ADScalar> ADActionModelAbstract;
  typedef crocoddyl::ActuationModelFullTpl<ADScalar> ADActuationModelFull;
  typedef crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<ADScalar> ADDifferentialActionModelFreeFwdDynamics;
  typedef crocoddyl::IntegratedActionModelEulerTpl<ADScalar> ADIntegratedActionModelEuler;

  pinocchio::ModelTpl<ADScalar> ad_model(model.cast<ADScalar>());
  boost::shared_ptr<crocoddyl::StateMultibodyTpl<ADScalar> > ad_state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<ADScalar> >(
          boost::make_shared<pinocchio::ModelTpl<ADScalar> >(ad_model));

  ADVectorXs ad_q0 = ADVectorXs::Random(ad_state->get_nq());
  ADVectorXs ad_x0(ad_state->get_nx());
  ad_x0 << ad_q0, ADVectorXs::Random(ad_state->get_nv());

  ADFramePlacement ad_Mref(
      ad_model.getFrameId("gripper_left_joint"),
      pinocchio::SE3Tpl<ADScalar>(ADMatrix3s::Identity(), ADVector3s((ADScalar)0, (ADScalar)0, (ADScalar).4)));
  boost::shared_ptr<ADCostModelAbstract> ad_goalTrackingCost =
      boost::make_shared<ADCostModelFramePlacement>(ad_state, ad_Mref);
  boost::shared_ptr<ADCostModelAbstract> ad_xRegCost = boost::make_shared<ADCostModelState>(ad_state);
  boost::shared_ptr<ADCostModelAbstract> ad_uRegCost = boost::make_shared<ADCostModelControl>(ad_state);

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<ADCostModelSum> ad_runningCostModel = boost::make_shared<ADCostModelSum>(ad_state);
  boost::shared_ptr<ADCostModelSum> ad_terminalCostModel = boost::make_shared<ADCostModelSum>(ad_state);

  // Then let's added the running and terminal cost functions
  ad_runningCostModel->addCost("gripperPose", ad_goalTrackingCost, ADScalar(1));
  ad_runningCostModel->addCost("xReg", ad_xRegCost, ADScalar(1e-4));
  ad_runningCostModel->addCost("uReg", ad_uRegCost, ADScalar(1e-4));
  ad_terminalCostModel->addCost("gripperPose", ad_goalTrackingCost, ADScalar(1));

  // We define an actuation model
  boost::shared_ptr<ADActuationModelFull> ad_actuation = boost::make_shared<ADActuationModelFull>(ad_state);

  // Next, we need to create an action model for running and terminal knots. The
  // forward dynamics (computed using ABA) are implemented
  // inside DifferentialActionModelFullyActuated.
  boost::shared_ptr<ADDifferentialActionModelFreeFwdDynamics> ad_runningDAM =
      boost::make_shared<ADDifferentialActionModelFreeFwdDynamics>(ad_state, ad_actuation, ad_runningCostModel);

  boost::shared_ptr<ADDifferentialActionModelFreeFwdDynamics> ad_terminalDAM =
      boost::make_shared<ADDifferentialActionModelFreeFwdDynamics>(ad_state, ad_actuation, ad_terminalCostModel);

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
      boost::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar> >(ad_runningModel, runningModel,
                                                                    "pyrene_model_running");
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > cg_terminalModel =
      boost::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar> >(ad_terminalModel, terminalModel,
                                                                    "pyrene_model_terminal");
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > > cg_runningModels(N, cg_runningModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> cg_problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, cg_runningModels, cg_terminalModel);
  for (unsigned int i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& cg_model = cg_problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> >& cg_data = cg_problem->get_runningDatas()[i];
    cg_model->quasiStatic(cg_data, us[i], x0);
  }

  crocoddyl::SolverDDP cg_ddp(cg_problem);
  cg_ddp.setCandidate(xs, us, false);

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
  assert_pretty(cg_runningData->xnext.isApprox(runningData->xnext), "Problem in xnext");
  assert_pretty(cg_runningData->cost == runningData->cost, "Problem in cost");
  assert_pretty(cg_runningData->Lx.isApprox(runningData->Lx), "Problem in Lx");
  assert_pretty(cg_runningData->Lu.isApprox(runningData->Lu), "Problem in Lu");
  assert_pretty(cg_runningData->Lxx.isApprox(runningData->Lxx), "Problem in Lxx");
  assert_pretty(cg_runningData->Lxu.isApprox(runningData->Lxu), "Problem in Lxu");
  assert_pretty(cg_runningData->Luu.isApprox(runningData->Luu), "Problem in Luu");
  assert_pretty(cg_runningData->Fx.isApprox(runningData->Fx), "Problem in Fx");
  assert_pretty(cg_runningData->Fu.isApprox(runningData->Fu), "Problem in Fu");
  /*****************************************************************************/

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
    // Timings pyrene-arm-calc+calcDiff
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
    // Timings pyrene-arm-calc+calcDiff
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

  // differential aba Timings
  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    duration_dcalcpin.setZero();
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
        boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> m =
            boost::static_pointer_cast<crocoddyl::IntegratedActionModelEuler>(problem->get_runningModels()[j]);
        boost::shared_ptr<crocoddyl::IntegratedActionDataEuler> d =
            boost::static_pointer_cast<crocoddyl::IntegratedActionDataEuler>(problem->get_runningDatas()[j]);

        boost::shared_ptr<crocoddyl::DifferentialActionDataFreeFwdDynamics> dd =
            boost::static_pointer_cast<crocoddyl::DifferentialActionDataFreeFwdDynamics>(d->differential);
        boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> dm =
            boost::static_pointer_cast<crocoddyl::DifferentialActionModelFreeFwdDynamics>(m->get_differential());

        Eigen::Ref<Eigen::VectorXd> q = xs[j].head(state->get_nq());
        Eigen::Ref<Eigen::VectorXd> v = xs[j].tail(state->get_nv());

        pinocchio::aba(dm->get_pinocchio(), dd->pinocchio, q, v, us[j]);
      }
      // end of calcdiff function
      duration_dcalcpin[i] = timer.get_duration();
    }
    avg_dcalcpin[ithread] = AVG(duration_dcalcpin);
    stddev_dcalcpin[ithread] = STDDEV(duration_dcalcpin);
    std::cout << ithread + 1 << " threaded aba [mean +- stddev in us]: " << avg_dcalcpin[ithread] << " +- "
              << stddev_dcalcpin[ithread] << " (per nodes/thread: " << avg_dcalcpin[ithread] * (ithread + 1) / N
              << " +- " << stddev_dcalcpin[ithread] * (ithread + 1) / N << ")" << std::endl;
  }

  // differential aba-derivatives Timings
  for (int ithread = 0; ithread < NTHREAD; ++ithread) {
    duration_diffcalcpin.setZero();
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
        boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> m =
            boost::static_pointer_cast<crocoddyl::IntegratedActionModelEuler>(problem->get_runningModels()[j]);
        boost::shared_ptr<crocoddyl::IntegratedActionDataEuler> d =
            boost::static_pointer_cast<crocoddyl::IntegratedActionDataEuler>(problem->get_runningDatas()[j]);

        boost::shared_ptr<crocoddyl::DifferentialActionDataFreeFwdDynamics> dd =
            boost::static_pointer_cast<crocoddyl::DifferentialActionDataFreeFwdDynamics>(d->differential);
        boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> dm =
            boost::static_pointer_cast<crocoddyl::DifferentialActionModelFreeFwdDynamics>(m->get_differential());

        Eigen::Ref<Eigen::VectorXd> q = xs[j].head(state->get_nq());
        Eigen::Ref<Eigen::VectorXd> v = xs[j].tail(state->get_nv());

        pinocchio::computeABADerivatives(dm->get_pinocchio(), dd->pinocchio, q, v, us[j]);
      }
      // end of calcdiff function
      duration_diffcalcpin[i] = timer.get_duration();
    }
    avg_diffcalcpin[ithread] = AVG(duration_diffcalcpin);
    stddev_diffcalcpin[ithread] = STDDEV(duration_diffcalcpin);
    std::cout << ithread + 1 << " threaded aba-derivs [mean +- stddev in us]: " << avg_diffcalcpin[ithread] << " +- "
              << stddev_diffcalcpin[ithread] << " (per nodes/thread: " << avg_diffcalcpin[ithread] * (ithread + 1) / N
              << " +- " << stddev_diffcalcpin[ithread] * (ithread + 1) / N << ")" << std::endl;
  }

  ddp.calcDiff();
  Eigen::ArrayXd duration_bp(T);
  // std::cout << "Starting timing backwardpass"<<std::endl;

  // Timings pyrene-arm-backwardPass
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

  // Timings pyrene-arm-forwardPass
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
    // Timings pyrene-arm-calc+calcDiff
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
