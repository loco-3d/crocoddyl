///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/core/solvers/box-qp.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

BoxQP::BoxQP(const std::size_t nx, const std::size_t maxiter, const double th_acceptstep, const double th_grad,
             const double reg)
    : nx_(nx),
      maxiter_(maxiter),
      th_acceptstep_(th_acceptstep),
      th_grad_(th_grad),
      reg_(reg),
      fold_(0.),
      fnew_(0.),
      x_(nx),
      xnew_(nx),
      g_(nx),
      dx_(nx) {
  // Check if values have a proper range
  if (0. >= th_acceptstep && th_acceptstep >= 0.5) {
    std::cerr << "Warning: th_acceptstep value should between 0 and 0.5" << std::endl;
  }
  if (0. > th_grad) {
    std::cerr << "Warning: th_grad value has to be positive." << std::endl;
  }
  if (0. > reg) {
    std::cerr << "Warning: reg value has to be positive." << std::endl;
  }

  // Initialized the values of vectors
  x_.setZero();
  xnew_.setZero();
  g_.setZero();
  dx_.setZero();

  // Reserve the space and compute alphas
  solution_.x = Eigen::VectorXd::Zero(nx);
  solution_.clamped_idx.reserve(nx_);
  solution_.free_idx.reserve(nx_);
  const std::size_t n_alphas_ = 10;
  alphas_.resize(n_alphas_);
  for (std::size_t n = 0; n < n_alphas_; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
}

BoxQP::~BoxQP() {}

const BoxQPSolution& BoxQP::solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& lb,
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
    solution_.clamped_idx.clear();
    solution_.free_idx.clear();
    // Compute the gradient
    g_ = q;
    g_.noalias() += H * x_;
    for (std::size_t j = 0; j < nx_; ++j) {
      const double gj = g_(j);
      const double xj = x_(j);
      const double lbj = lb(j);
      const double ubj = ub(j);
      if ((xj == lbj && gj > 0.) || (xj == ubj && gj < 0.)) {
        solution_.clamped_idx.push_back(j);
      } else {
        solution_.free_idx.push_back(j);
      }
    }

    // Check convergence
    nf_ = solution_.free_idx.size();
    nc_ = solution_.clamped_idx.size();
    if (g_.lpNorm<Eigen::Infinity>() <= th_grad_ || nf_ == 0) {
      if (k == 0) {  // compute the inverse of the free Hessian
        Hff_.resize(nf_, nf_);
        for (std::size_t i = 0; i < nf_; ++i) {
          const std::size_t fi = solution_.free_idx[i];
          for (std::size_t j = 0; j < nf_; ++j) {
            Hff_(i, j) = H(fi, solution_.free_idx[j]);
          }
        }
        if (reg_ != 0.) {
          Hff_.diagonal().array() += reg_;
        }
        Hff_inv_llt_.compute(Hff_);
        const Eigen::ComputationInfo& info = Hff_inv_llt_.info();
        if (info != Eigen::Success) {
          throw_pretty("backward_error");
        }
        solution_.Hff_inv.setIdentity(nf_, nf_);
        Hff_inv_llt_.solveInPlace(solution_.Hff_inv);
      }
      solution_.x = x_;
      return solution_;
    }

    // Compute the search direction as Newton step along the free space
    qf_.resize(nf_);
    xf_.resize(nf_);
    xc_.resize(nc_);
    dxf_.resize(nf_);
    Hff_.resize(nf_, nf_);
    Hfc_.resize(nf_, nc_);
    for (std::size_t i = 0; i < nf_; ++i) {
      const std::size_t fi = solution_.free_idx[i];
      qf_(i) = q(fi);
      xf_(i) = x_(fi);
      for (std::size_t j = 0; j < nf_; ++j) {
        Hff_(i, j) = H(fi, solution_.free_idx[j]);
      }
      for (std::size_t j = 0; j < nc_; ++j) {
        const std::size_t cj = solution_.clamped_idx[j];
        xc_(j) = x_(cj);
        Hfc_(i, j) = H(fi, cj);
      }
    }
    if (reg_ != 0.) {
      Hff_.diagonal().array() += reg_;
    }
    Hff_inv_llt_.compute(Hff_);
    const Eigen::ComputationInfo& info = Hff_inv_llt_.info();
    if (info != Eigen::Success) {
      throw_pretty("backward_error");
    }
    solution_.Hff_inv.setIdentity(nf_, nf_);
    Hff_inv_llt_.solveInPlace(solution_.Hff_inv);
    dxf_ = -qf_;
    if (nc_ != 0) {
      dxf_.noalias() -= Hfc_ * xc_;
    }
    Hff_inv_llt_.solveInPlace(dxf_);
    dxf_ -= xf_;
    dx_.setZero();
    for (std::size_t i = 0; i < nf_; ++i) {
      dx_(solution_.free_idx[i]) = dxf_(i);
    }

    // Try different step lengths
    fold_ = 0.5 * x_.dot(H * x_) + q.dot(x_);
    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      double steplength = *it;
      for (std::size_t i = 0; i < nx_; ++i) {
        xnew_(i) = std::max(std::min(x_(i) + steplength * dx_(i), ub(i)), lb(i));
      }
      fnew_ = 0.5 * xnew_.dot(H * xnew_) + q.dot(xnew_);
      if (fold_ - fnew_ > th_acceptstep_ * g_.dot(x_ - xnew_)) {
        x_ = xnew_;
        break;
      }
    }
  }
  solution_.x = x_;
  return solution_;
}

const BoxQPSolution& BoxQP::get_solution() const { return solution_; }

std::size_t BoxQP::get_nx() const { return nx_; }

std::size_t BoxQP::get_maxiter() const { return maxiter_; }

double BoxQP::get_th_acceptstep() const { return th_acceptstep_; }

double BoxQP::get_th_grad() const { return th_grad_; }

double BoxQP::get_reg() const { return reg_; }

const std::vector<double>& BoxQP::get_alphas() const { return alphas_; }

void BoxQP::set_nx(const std::size_t nx) {
  nx_ = nx;
  x_ = Eigen::VectorXd::Zero(nx);
  xnew_ = Eigen::VectorXd::Zero(nx);
  g_ = Eigen::VectorXd::Zero(nx);
  dx_ = Eigen::VectorXd::Zero(nx);
}

void BoxQP::set_maxiter(const std::size_t maxiter) { maxiter_ = maxiter; }

void BoxQP::set_th_acceptstep(const double th_acceptstep) {
  if (0. >= th_acceptstep && th_acceptstep >= 0.5) {
    throw_pretty("Invalid argument: "
                 << "th_acceptstep value should between 0 and 0.5");
  }
  th_acceptstep_ = th_acceptstep;
}

void BoxQP::set_th_grad(const double th_grad) {
  if (0. > th_grad) {
    throw_pretty("Invalid argument: "
                 << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

void BoxQP::set_reg(const double reg) {
  if (0. > reg) {
    throw_pretty("Invalid argument: "
                 << "reg value has to be positive.");
  }
  reg_ = reg;
}

void BoxQP::set_alphas(const std::vector<double>& alphas) {
  double prev_alpha = alphas[0];
  if (prev_alpha != 1.) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    double alpha = alphas[i];
    if (0. >= alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
}

}  // namespace crocoddyl
