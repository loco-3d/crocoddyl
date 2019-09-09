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

int main() {
  unsigned int NX = 37;
  unsigned int NU = 12;
  bool CALLBACKS = false;
  unsigned int N = 100;  // number of nodes
  unsigned int T = 5e3;  // number of trials
  unsigned int MAXITER = 1;
  using namespace crocoddyl;

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
  std::clock_t c_start, c_end;
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    c_start = std::clock();
    ddp.solve(xs, us, MAXITER);
    c_end = std::clock();
    duration[i] = 1e3 * (double)(c_end - c_start) / CLOCKS_PER_SEC;
  }

  double avrg_duration = duration.sum() / T;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "CPU time [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")" << std::endl;
}