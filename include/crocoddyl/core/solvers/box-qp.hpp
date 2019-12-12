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

#include "crocoddyl/core/utils/exception.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <vector>

namespace crocoddyl {

struct BoxQPSolution {
  BoxQPSolution(const Eigen::MatrixXd& Hff_inv, const Eigen::VectorXd& x, const std::vector<size_t>& free_idx,
                const std::vector<size_t>& clamped_idx)
      : Hff_inv(Hff_inv), x(x), free_idx(free_idx), clamped_idx(clamped_idx) {}

  Eigen::MatrixXd Hff_inv;
  Eigen::MatrixXd x;
  std::vector<size_t> free_idx;
  std::vector<size_t> clamped_idx;
};

// Based on Yuval Tassa's BoxQP
// Cf. https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
inline BoxQPSolution BoxQP(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& lb,
                           const Eigen::VectorXd& ub, const Eigen::VectorXd& xinit, const std::size_t maxiter = 100,
                           const double th_acceptstep = 0.1, const double th_grad = 1e-9, const double reg = 1e-9) {
  const std::size_t nx = xinit.size();
  if (static_cast<std::size_t>(H.rows()) != nx || static_cast<std::size_t>(H.cols()) != nx) {
    throw_pretty("Invalid argument: "
                 << "H has wrong dimension (it should be " + std::to_string(nx) + "," + std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(q.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "q has wrong dimension (it should be " + std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(lb.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "lb has wrong dimension (it should be " + std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(ub.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "ub has wrong dimension (it should be " + std::to_string(nx) + ")");
  }

  std::vector<double> alphas;
  const std::size_t& n_alphas = 10;
  alphas.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas[n] = 1. / pow(2., static_cast<double>(n));
  }

  // We need to enforce feasible warm-starting of the algorithm
  Eigen::VectorXd x(nx), xnew(nx), g(nx);
  for (std::size_t i = 0; i < nx; ++i) {
    x(i) = std::max(std::min(xinit(i), ub(i)), lb(i));
  }

  // Start the numerical iterations
  Eigen::VectorXd qf, xf, xc, dx, dxf;
  Eigen::MatrixXd Hff, Hff_inv, Hfc;
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt;
  std::vector<size_t> clamped_idx, free_idx;
  for (std::size_t k = 0; k < maxiter; ++k) {
    clamped_idx.clear();
    free_idx.clear();
    // Compute the gradient
    g = q;
    g.noalias() += H * x;
    for (std::size_t j = 0; j < nx; ++j) {
      const double& gj = g(j);
      const double& xj = x(j);
      const double& lbj = lb(j);
      const double& ubj = ub(j);
      if ((xj == lbj && gj > 0.) || (xj == ubj && gj < 0.)) {
        clamped_idx.push_back(j);
      } else {
        free_idx.push_back(j);
      }
    }

    // Check convergence
    const std::size_t& nc = clamped_idx.size();
    const std::size_t& nf = free_idx.size();
    if (g.lpNorm<Eigen::Infinity>() <= th_grad || nf == 0) {
      if (k == 0) {  // compute the inverse of the free Hessian
        Hff.resize(nf, nf);
        for (std::size_t i = 0; i < nf; ++i) {
          for (std::size_t j = 0; j < nf; ++j) {
            Hff(i, j) = H(free_idx[i], free_idx[j]);
          }
        }
        Hff_inv_llt = Eigen::LLT<Eigen::MatrixXd>(nf);
        Hff_inv_llt.compute(Eigen::MatrixXd::Identity(nf, nf) * reg + Hff);
        Eigen::ComputationInfo info = Hff_inv_llt.info();
        if (info != Eigen::Success) {
          throw_pretty("backward_error");
        }
        Hff_inv.setIdentity(nf, nf);
        Hff_inv_llt.solveInPlace(Hff_inv);
      }
      return BoxQPSolution(Hff_inv, x, free_idx, clamped_idx);
    }

    // Compute the search direaction as Newton step along the free space
    qf.resize(nf);
    xf.resize(nf);
    xc.resize(nc);
    dx.resize(nx);
    dxf.resize(nf);
    Hff.resize(nf, nf);
    Hfc.resize(nf, nc);
    for (std::size_t i = 0; i < nf; ++i) {
      qf(i) = q(free_idx[i]);
      xf(i) = x(free_idx[i]);
      for (std::size_t j = 0; j < nf; ++j) {
        Hff(i, j) = H(free_idx[i], free_idx[j]);
      }
      for (std::size_t j = 0; j < nc; ++j) {
        xc(j) = x(clamped_idx[j]);
        Hfc(i, j) = H(free_idx[i], clamped_idx[j]);
      }
    }
    Hff_inv_llt = Eigen::LLT<Eigen::MatrixXd>(nf);
    Hff_inv_llt.compute(Hff);
    Eigen::ComputationInfo info = Hff_inv_llt.info();
    if (info != Eigen::Success) {
      std::cout << "hey" << std::endl;
      throw_pretty("backward_error");
    }
    Hff_inv.setIdentity(nf, nf);
    Hff_inv_llt.solveInPlace(Hff_inv);
    dxf = -qf;
    if (nc != 0) {
      dxf.noalias() -= Hfc * xc;
    }
    Hff_inv_llt.solveInPlace(dxf);
    dxf -= xf;
    dx.setZero();
    for (std::size_t i = 0; i < nf; ++i) {
      dx(free_idx[i]) = dxf(i);
    }

    // There is not improvement anymore
    if (dx.lpNorm<Eigen::Infinity>() < th_grad) {
      return BoxQPSolution(Hff_inv, x, free_idx, clamped_idx);
    }

    // Try different step lengths
    double fold = 0.5 * x.dot(H * x) + q.dot(x);
    for (std::vector<double>::const_iterator it = alphas.begin(); it != alphas.end(); ++it) {
      double steplength = *it;

      for (std::size_t i = 0; i < nx; ++i) {
        xnew(i) = std::max(std::min(x(i) + steplength * dx(i), ub(i)), lb(i));
      }
      double fnew = 0.5 * xnew.dot(H * xnew) + q.dot(xnew);
      if (fold - fnew > th_acceptstep * g.dot(x - xnew)) {
        x = xnew;
        break;
      }
    }
  }
  return BoxQPSolution(Hff_inv, x, free_idx, clamped_idx);
}

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
