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

#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/core/codegen/action-base.hpp"
#include "factory/biped.hpp"

#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / (static_cast<double>(vec.size()) - 1))
#define AVG(vec) (vec.mean())

int main(int argc, char* argv[]) {
  unsigned int N = 100;  // number of nodes
  unsigned int T = 1e3;  // number of trials
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  // Building the running and terminal models
  typedef CppAD::AD<CppAD::cg::CG<double> > ADScalar;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel, terminalModel;
  crocoddyl::benchmark::build_biped_action_models(runningModel, terminalModel);

  // Code generation of the running an terminal models
  boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar> > ad_runningModel, ad_terminalModel;
  crocoddyl::benchmark::build_biped_action_models(ad_runningModel, ad_terminalModel);
  boost::shared_ptr<crocoddyl::ActionModelAbstract> cg_runningModel =
      boost::make_shared<crocoddyl::ActionModelCodeGen>(ad_runningModel, runningModel,
                                                        "biped_with_contact_running_cg");
  boost::shared_ptr<crocoddyl::ActionModelAbstract> cg_terminalModel =
      boost::make_shared<crocoddyl::ActionModelCodeGen>(ad_terminalModel, terminalModel,
                                                        "biped_with_contact_terminal_cg");

  // Get the initial state
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(runningModel->get_state());
  std::cout << "NQ: " << state->get_nq() << std::endl;
  std::cout << "Number of nodes: " << N << std::endl << std::endl;
  Eigen::VectorXd x0(state->rand());

  // Defining the shooting problem for both cases: with and without code generation
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels(N, runningModel);
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > cg_runningModels(N, cg_runningModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, terminalModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> cg_problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, cg_runningModels, cg_terminalModel);

  // Computing the warm-start
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(runningModel->get_nu()));
  for (unsigned int i = 0; i < N; ++i) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem->get_runningDatas()[i];
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& cg_model = cg_problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& cg_data = cg_problem->get_runningDatas()[i];
    model->quasiStatic(data, us[i], x0);
    cg_model->quasiStatic(cg_data, us[i], x0);
  }

  // Check that code-generated action model is the same as original.
  /**************************************************************************/
  crocoddyl::SolverDDP ddp(problem);
  crocoddyl::SolverDDP cg_ddp(cg_problem);
  ddp.setCandidate(xs, us, false);
  cg_ddp.setCandidate(xs, us, false);
  boost::shared_ptr<crocoddyl::ActionDataAbstract> cg_runningData = cg_runningModel->createData();
  boost::shared_ptr<crocoddyl::ActionDataAbstract> runningData = runningModel->createData();
  Eigen::VectorXd x_rand = cg_runningModel->get_state()->rand();
  Eigen::VectorXd u_rand = Eigen::VectorXd::Random(cg_runningModel->get_nu());
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
    ddp.forwardPass(0.005);
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
