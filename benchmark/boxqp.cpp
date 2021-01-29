///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/box-qp.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  unsigned int NX = 36; // dimension of the decision vector
  unsigned int T = 5e3; // number of trials
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  // Solving the bounded QP problem
  crocoddyl::BoxQP boxqp(NX);
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    // Creating a new random problem
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(NX, NX);
    Eigen::MatrixXd hessian = H.transpose() * H;
    hessian = 0.5 * (hessian + hessian.transpose()).eval();
    Eigen::VectorXd gradient = Eigen::VectorXd::Random(NX);
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(NX);
    Eigen::VectorXd ub = Eigen::VectorXd::Ones(NX);
    Eigen::VectorXd xinit(NX);

    crocoddyl::Timer timer;
    boxqp.solve(hessian, gradient, lb, ub, xinit);
    duration[i] = timer.get_duration();
  }

  double avrg_duration = duration.sum() / T;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  BoxQP.solve (36) [ms]: " << avrg_duration << " ("
            << min_duration << "-" << max_duration << ")" << std::endl;

  NX = 76;
  boxqp.set_nx(NX);
  for (unsigned int i = 0; i < T; ++i) {
    // Creating a new random problem
    Eigen::MatrixXd H = Eigen::MatrixXd::Random(NX, NX);
    Eigen::MatrixXd hessian = H.transpose() * H;
    hessian = 0.5 * (hessian + hessian.transpose()).eval();
    Eigen::VectorXd gradient = Eigen::VectorXd::Random(NX);
    Eigen::VectorXd lb = Eigen::VectorXd::Zero(NX);
    Eigen::VectorXd ub = Eigen::VectorXd::Ones(NX);
    Eigen::VectorXd xinit(NX);

    crocoddyl::Timer timer;
    boxqp.solve(hessian, gradient, lb, ub, xinit);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  BoxQP.solve (76) [ms]: " << avrg_duration << " ("
            << min_duration << "-" << max_duration << ")" << std::endl;
}