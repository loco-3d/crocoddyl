///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include <time.h>

#ifdef WITH_MULTITHREADING
#include <omp.h>
#endif  // WITH_MULTITHREADING

int main(int argc, char* argv[]) {
  bool CALLBACKS = false;
  unsigned int N = 200;  // number of nodes
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  // Creating the action models and warm point for the unicycle system
  Eigen::VectorXd x0 = Eigen::Vector3d(1., 0., 0.);
  crocoddyl::ActionModelAbstract* model = new crocoddyl::ActionModelUnicycle();
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::Vector2d::Zero());
  std::vector<crocoddyl::ActionModelAbstract*> runningModels(N, model);

  // Formulating the optimal control problem
  crocoddyl::ShootingProblem problem(x0, runningModels, model);
  crocoddyl::SolverDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<crocoddyl::CallbackAbstract*> cbs;
    cbs.push_back(new crocoddyl::CallbackVerbose());
    ddp.setCallbacks(cbs);
  }

  // Solving the optimal control problem
  struct timespec start, finish;
  double elapsed;
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    ddp.solve(xs, us, MAXITER);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
  }

  double avrg_duration = duration.sum() / T;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  // Running calc
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    problem.calc(xs, us);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;

  // Running calcDiff
  for (unsigned int i = 0; i < T; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &start);
    problem.calcDiff(xs, us);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = static_cast<double>(finish.tv_sec - start.tv_sec) * 1000000;
    elapsed += static_cast<double>(finish.tv_nsec - start.tv_nsec) / 1000;
    duration[i] = elapsed / 1000.;
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration
            << ")" << std::endl;
}
