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
  BoxQPSolution() {}
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
class BoxQP {
 public:
  BoxQP(const std::size_t nx, std::size_t maxiter = 100, const double th_acceptstep = 0.1, const double th_grad = 1e-9,
        const double reg = 1e-9)
      : nx_(nx),
        maxiter_(maxiter),
        th_acceptstep_(th_acceptstep),
        th_grad_(th_grad),
        reg_(reg),
        x_(nx),
        xnew_(nx),
        g_(nx),
        dx_(nx) {
    const std::size_t& n_alphas_ = 10;
    alphas_.resize(n_alphas_);
    for (std::size_t n = 0; n < n_alphas_; ++n) {
      alphas_[n] = 1. / pow(2., static_cast<double>(n));
    }
  }
  ~BoxQP() {}

  const BoxQPSolution& solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& lb,
                             const Eigen::VectorXd& ub, const Eigen::VectorXd& xinit) {
    if (static_cast<std::size_t>(H.rows()) != nx_ || static_cast<std::size_t>(H.cols()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "H has wrong dimension (it should be " + std::to_string(nx_) + "," + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(q.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "q has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(lb.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "lb has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(ub.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "ub has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(xinit.size()) != nx_) {
      throw_pretty("Invalid argument: "
                   << "xinit has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }

    // We need to enforce feasible warm-starting of the algorithm
    for (std::size_t i = 0; i < nx_; ++i) {
      x_(i) = std::max(std::min(xinit(i), ub(i)), lb(i));
    }

    // Start the numerical iterations
    for (std::size_t k = 0; k < maxiter_; ++k) {
      clamped_idx_.clear();
      free_idx_.clear();
      // Compute the gradient
      g_ = q;
      g_.noalias() += H * x_;
      for (std::size_t j = 0; j < nx_; ++j) {
        const double& gj = g_(j);
        const double& xj = x_(j);
        const double& lbj = lb(j);
        const double& ubj = ub(j);
        if ((xj == lbj && gj > 0.) || (xj == ubj && gj < 0.)) {
          clamped_idx_.push_back(j);
        } else {
          free_idx_.push_back(j);
        }
      }

      // Check convergence
      nf_ = free_idx_.size();
      nc_ = clamped_idx_.size();
      if (g_.lpNorm<Eigen::Infinity>() <= th_grad_ || nf_ == 0) {
        if (k == 0) {  // compute the inverse of the free Hessian
          Hff_.resize(nf_, nf_);
          for (std::size_t i = 0; i < nf_; ++i) {
            for (std::size_t j = 0; j < nf_; ++j) {
              Hff_(i, j) = H(free_idx_[i], free_idx_[j]);
            }
          }
          Hff_inv_llt_ = Eigen::LLT<Eigen::MatrixXd>(nf_);
          Hff_inv_llt_.compute(Eigen::MatrixXd::Identity(nf_, nf_) * reg_ + Hff_);
          Eigen::ComputationInfo info = Hff_inv_llt_.info();
          if (info != Eigen::Success) {
            throw_pretty("backward_error");
          }
          Hff_inv_.setIdentity(nf_, nf_);
          Hff_inv_llt_.solveInPlace(Hff_inv_);
        }
        solution_.Hff_inv = Hff_inv_;
        solution_.x = x_;
        solution_.free_idx = free_idx_;
        solution_.clamped_idx = clamped_idx_;
        return solution_;
      }

      // Compute the search direaction as Newton step along the free space
      qf_.resize(nf_);
      xf_.resize(nf_);
      xc_.resize(nc_);
      dxf_.resize(nf_);
      Hff_.resize(nf_, nf_);
      Hfc_.resize(nf_, nc_);
      for (std::size_t i = 0; i < nf_; ++i) {
        qf_(i) = q(free_idx_[i]);
        xf_(i) = x_(free_idx_[i]);
        for (std::size_t j = 0; j < nf_; ++j) {
          Hff_(i, j) = H(free_idx_[i], free_idx_[j]);
        }
        for (std::size_t j = 0; j < nc_; ++j) {
          xc_(j) = x_(clamped_idx_[j]);
          Hfc_(i, j) = H(free_idx_[i], clamped_idx_[j]);
        }
      }
      Hff_inv_llt_ = Eigen::LLT<Eigen::MatrixXd>(nf_);
      Hff_inv_llt_.compute(Hff_);
      Eigen::ComputationInfo info = Hff_inv_llt_.info();
      if (info != Eigen::Success) {
        throw_pretty("backward_error");
      }
      Hff_inv_.setIdentity(nf_, nf_);
      Hff_inv_llt_.solveInPlace(Hff_inv_);
      dxf_ = -qf_;
      if (nc_ != 0) {
        dxf_.noalias() -= Hfc_ * xc_;
      }
      Hff_inv_llt_.solveInPlace(dxf_);
      dxf_ -= xf_;
      dx_.setZero();
      for (std::size_t i = 0; i < nf_; ++i) {
        dx_(free_idx_[i]) = dxf_(i);
      }

      // There is not improvement anymore
      if (dx_.lpNorm<Eigen::Infinity>() < th_grad_) {
        solution_.Hff_inv = Hff_inv_;
        solution_.x = x_;
        solution_.free_idx = free_idx_;
        solution_.clamped_idx = clamped_idx_;
        return solution_;
      }

      // Try different step lengths
      double fold = 0.5 * x_.dot(H * x_) + q.dot(x_);
      for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
        double steplength = *it;
        for (std::size_t i = 0; i < nx_; ++i) {
          xnew_(i) = std::max(std::min(x_(i) + steplength * dx_(i), ub(i)), lb(i));
        }
        double fnew = 0.5 * xnew_.dot(H * xnew_) + q.dot(xnew_);
        if (fold - fnew > th_acceptstep_ * g_.dot(x_ - xnew_)) {
          x_ = xnew_;
          break;
        }
      }
    }
    solution_.Hff_inv = Hff_inv_;
    solution_.x = x_;
    solution_.free_idx = free_idx_;
    solution_.clamped_idx = clamped_idx_;
    return solution_;
  }

 private:
  std::size_t nx_;
  BoxQPSolution solution_;
  std::size_t maxiter_;
  double th_acceptstep_;
  double th_grad_;
  double reg_;

  std::size_t nf_;
  std::size_t nc_;
  std::vector<double> alphas_;
  Eigen::VectorXd x_;
  Eigen::VectorXd xnew_;
  Eigen::VectorXd g_;
  Eigen::VectorXd dx_;

  Eigen::VectorXd qf_;
  Eigen::VectorXd xf_;
  Eigen::VectorXd xc_;
  Eigen::VectorXd dxf_;
  Eigen::MatrixXd Hff_;
  Eigen::MatrixXd Hff_inv_;
  Eigen::MatrixXd Hfc_;
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt_;
  std::vector<size_t> clamped_idx_;
  std::vector<size_t> free_idx_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
