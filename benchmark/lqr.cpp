///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include <ctime>

int main(int argc, char* argv[]) {
  unsigned int NX = 37;
  unsigned int NU = 12;
  bool CALLBACKS = false;
  unsigned int N = 100;  // number of nodes
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  using namespace crocoddyl;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<ActionModelAbstract*> runningModels;
  ActionModelAbstract* terminalModel;
  x0 = Eigen::VectorXd::Zero(NX);

  // Creating the action models and warm point for the LQR system
  for (unsigned int i = 0; i < N; ++i) {
    ActionModelAbstract* model_i = new ActionModelLQR(NX, NU);
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::VectorXd::Zero(NU));
  }
  xs.push_back(x0);
  terminalModel = new ActionModelLQR(NX, NU);

  // Formulating the optimal control problem
  ShootingProblem problem(x0, runningModels, terminalModel);
  SolverDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<CallbackAbstract*> cbs;
    cbs.push_back(new CallbackVerbose());
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
  std::cout << "DDP.solve [mu]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")" << std::endl;

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
  std::cout << "ShootingProblem.calc [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
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
  std::cout << "ShootingProblem.calcDiff [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
            << std::endl;
}