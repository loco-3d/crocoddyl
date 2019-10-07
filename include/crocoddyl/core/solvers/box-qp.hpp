///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
//
// This file was originally part of Exotica, cf.
// https://github.com/ipab-slmc/exotica/blob/master/exotica_core/include/exotica_core/tools/box_qp.h
//
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <vector>

namespace crocoddyl {
struct BoxQPSolution {
  BoxQPSolution(const Eigen::MatrixXd& Hff_inv_in, const Eigen::VectorXd& x_in, const std::vector<size_t>& free_idx_in,
                const std::vector<size_t>& clamped_idx_in)
      : Hff_inv(Hff_inv_in), x(x_in), free_idx(free_idx_in), clamped_idx(clamped_idx_in) {}

  Eigen::MatrixXd Hff_inv;
  Eigen::MatrixXd x;
  std::vector<size_t> free_idx;
  std::vector<size_t> clamped_idx;
};

// Based on Yuval Tassa's BoxQP
// Cf. https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
inline BoxQPSolution BoxQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& b_low,
                           const Eigen::VectorXd& b_high, const Eigen::VectorXd& x_init, const double gamma,
                           const int max_iterations, const double epsilon, const double lambda) {
  if (max_iterations < 0) {
    throw std::runtime_error("Max iterations needs to be positive.");
  }

  int it = 0;
  Eigen::VectorXd delta_xf(x_init.size()), x = x_init;
  std::vector<size_t> clamped_idx, free_idx;
  Eigen::VectorXd grad = q + H * x_init;
  Eigen::MatrixXd Hff, Hfc, Hff_inv;
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt = Eigen::LLT<Eigen::MatrixXd>(H.rows());

  Hff_inv_llt.compute(Eigen::MatrixXd::Identity(H.rows(), H.cols()) * lambda + H);
  Hff_inv_llt.solveInPlace(Hff_inv);

  if (grad.lpNorm<Eigen::Infinity>() <= epsilon) {
    return BoxQPSolution(Hff_inv, x_init, free_idx, clamped_idx);
  }

  while (grad.lpNorm<Eigen::Infinity>() > epsilon && it < max_iterations) {
    ++it;
    grad.noalias() = q + H * x;
    clamped_idx.clear();
    free_idx.clear();

    for (int i = 0; i < grad.size(); ++i) {
      if ((x(i) == b_low(i) && grad(i) > 0) || (x(i) == b_high(i) && grad(i) < 0)) {
        clamped_idx.push_back(i);
      } else {
        free_idx.push_back(i);
      }
    }

    if (free_idx.size() == 0) {
      return BoxQPSolution(Hff_inv, x, free_idx, clamped_idx);
    }

    Hff.resize(free_idx.size(), free_idx.size());
    Hfc.resize(free_idx.size(), clamped_idx.size());

    if (clamped_idx.size() == 0) {
      Hff = H;
    } else {
      for (size_t i = 0; i < free_idx.size(); ++i) {
        for (size_t j = 0; j < free_idx.size(); ++j) {
          Hff(i, j) = H(free_idx[i], free_idx[j]);
        }
      }

      for (size_t i = 0; i < free_idx.size(); ++i) {
        for (size_t j = 0; j < clamped_idx.size(); ++j) {
          Hfc(i, j) = H(free_idx[i], clamped_idx[j]);
        }
      }
    }

    // NOTE: Array indexing not supported in current eigen version
    Eigen::VectorXd q_free(free_idx.size()), x_free(free_idx.size()), x_clamped(clamped_idx.size());
    for (size_t i = 0; i < free_idx.size(); ++i) {
      q_free(i) = q(free_idx[i]);
      x_free(i) = x(free_idx[i]);
    }

    for (size_t j = 0; j < clamped_idx.size(); ++j) {
      x_clamped(j) = x(clamped_idx[j]);
    }

    // The dimension of Hff has changed - reinitialise LLT
    // Hff_inv_llt = Eigen::LLT<Eigen::MatrixXd>(Hff.rows());
    // Hff_inv_llt.compute(Eigen::MatrixXd::Identity(Hff.rows(), Hff.cols()) * lambda + Hff);
    // Hff_inv_llt.solveInPlace(Hff_inv);
    // TODO: Use Cholesky, however, often unstable without adapting lambda.
    Hff_inv = (Eigen::MatrixXd::Identity(Hff.rows(), Hff.cols()) * lambda + Hff).inverse();

    if (clamped_idx.size() == 0) {
      // std::cout << "Hff_inv=" << Hff_inv.rows() << "x" << Hff_inv.cols() << ", q_free=" << q_free.size() << ",
      // x_free=" << x_free.size() << std::endl;
      delta_xf.noalias() = -Hff_inv * (q_free)-x_free;
    } else {
      delta_xf.noalias() = -Hff_inv * (q_free + Hfc * x_clamped) - x_free;
    }

    double f_old = (0.5 * x.transpose() * H * x + q.transpose() * x)(0);
    static const Eigen::VectorXd alpha_space = Eigen::VectorXd::LinSpaced(10, 1.0, 0.1);

    bool armijo_reached = false;
    Eigen::VectorXd x_new(x.size()), x_diff(x.size());
    for (int ai = 0; ai < alpha_space.rows(); ++ai) {
      x_new = x;
      for (size_t i = 0; i < free_idx.size(); ++i) {
        x_new(free_idx[i]) = std::max(std::min(x(free_idx[i]) + alpha_space[ai] * delta_xf(i), b_high(i)), b_low(i));
      }

      double f_new = (0.5 * x_new.transpose() * H * x_new + q.transpose() * x_new)(0);
      x_diff.noalias() = x - x_new;

      // armijo criterion>
      double armijo_coef = (f_old - f_new) / (grad.transpose() * x_diff + 1e-5);
      if (armijo_coef > gamma) {
        armijo_reached = true;
        x = x_new;
        break;
      }
    }

    // break if no step made
    if (!armijo_reached) break;
  }

  return BoxQPSolution(Hff_inv, x, free_idx, clamped_idx);
}

inline BoxQPSolution BoxQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& b_low,
                           const Eigen::VectorXd& b_high, const Eigen::VectorXd& x_init) {
  const double epsilon = 1e-5;
  const double gamma = 0.1;
  const int max_iterations = 100;
  const double lambda = 1e-5;
  return BoxQP(H, q, b_low, b_high, x_init, gamma, max_iterations, epsilon, lambda);
}
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
