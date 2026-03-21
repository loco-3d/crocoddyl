///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
BoxQPTpl<Scalar>::BoxQPTpl(const std::size_t nx, const std::size_t maxiter,
                           const Scalar th_acceptstep, const Scalar th_grad,
                           const Scalar reg)
    : nx_(nx),
      maxiter_(maxiter),
      th_acceptstep_(th_acceptstep),
      th_grad_(th_grad),
      reg_(reg),
      fold_(Scalar(0.)),
      fnew_(Scalar(0.)),
      x_(nx),
      xnew_(nx),
      g_(nx),
      dx_(nx),
      xo_(nx),
      dxo_(nx),
      qo_(nx),
      Ho_(nx, nx) {
  // Check if values have a proper range
  if (Scalar(0.) >= th_acceptstep && th_acceptstep >= Scalar(0.5)) {
    std::cerr << "Warning: th_acceptstep value should between 0 and 0.5"
              << std::endl;
  }
  if (Scalar(0.) > th_grad) {
    std::cerr << "Warning: th_grad value has to be positive." << std::endl;
  }
  if (Scalar(0.) > reg) {
    std::cerr << "Warning: reg value has to be positive." << std::endl;
  }
  // Initialized the values of vectors
  x_.setZero();
  xnew_.setZero();
  g_.setZero();
  dx_.setZero();
  xo_.setZero();
  dxo_.setZero();
  qo_.setZero();
  Ho_.setZero();
  // Reserve the space and compute alphas
  solution_.x = VectorXs::Zero(nx);
  solution_.clamped_idx.reserve(nx_);
  solution_.free_idx.reserve(nx_);
  const std::size_t n_alphas_ = 10;
  alphas_.resize(n_alphas_);
  for (std::size_t n = 0; n < n_alphas_; ++n) {
    alphas_[n] = Scalar(1.) / pow(Scalar(2.), static_cast<Scalar>(n));
  }
}

template <typename Scalar>
const BoxQPSolutionTpl<Scalar>& BoxQPTpl<Scalar>::solve(const MatrixXs& H,
                                                        const VectorXs& q,
                                                        const VectorXs& lb,
                                                        const VectorXs& ub,
                                                        const VectorXs& xinit) {
  if (static_cast<std::size_t>(H.rows()) != nx_ ||
      static_cast<std::size_t>(H.cols()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "H has wrong dimension (it should be " +
                        std::to_string(nx_) + "," + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(q.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "q has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(lb.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "lb has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(ub.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "ub has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(xinit.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "xinit has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  // We need to enforce feasible warm-starting of the algorithm
  for (std::size_t i = 0; i < nx_; ++i) {
    x_(i) = std::max(std::min(xinit(i), ub(i)), lb(i));
  }
  // Start the numerical iterations
  for (std::size_t k = 0; k < maxiter_; ++k) {
    solution_.clamped_idx.clear();
    solution_.free_idx.clear();
    // Compute the Cauchy point and active set
    g_ = q;
    g_.noalias() += H * x_;
    if (reg_ != Scalar(0.)) {
      g_.noalias() += reg_ * x_;
    }
    for (std::size_t j = 0; j < nx_; ++j) {
      const Scalar gj = g_(j);
      const Scalar xj = x_(j);
      const Scalar lbj = lb(j);
      const Scalar ubj = ub(j);
      if ((xj == lbj && gj > Scalar(0.)) || (xj == ubj && gj < Scalar(0.))) {
        solution_.clamped_idx.push_back(j);
      } else {
        solution_.free_idx.push_back(j);
      }
    }
    // Compute the search direction as Newton step along the free space
    nf_ = solution_.free_idx.size();
    nc_ = solution_.clamped_idx.size();
    Eigen::VectorBlock<VectorXs> xf = xo_.head(nf_);
    Eigen::VectorBlock<VectorXs> xc = xo_.tail(nc_);
    Eigen::VectorBlock<VectorXs> dxf = dxo_.head(nf_);
    Eigen::VectorBlock<VectorXs> qf = qo_.head(nf_);
    Eigen::Block<MatrixXs> Hff = Ho_.topLeftCorner(nf_, nf_);
    Eigen::Block<MatrixXs> Hfc = Ho_.topRightCorner(nf_, nc_);
    for (std::size_t i = 0; i < nf_; ++i) {
      const std::size_t fi = solution_.free_idx[i];
      qf(i) = q(fi);
      xf(i) = x_(fi);
      for (std::size_t j = 0; j < nf_; ++j) {
        Hff(i, j) = H(fi, solution_.free_idx[j]);
      }
      for (std::size_t j = 0; j < nc_; ++j) {
        const std::size_t cj = solution_.clamped_idx[j];
        xc(j) = x_(cj);
        Hfc(i, j) = H(fi, cj);
      }
    }
    if (reg_ != Scalar(0.)) {
      Hff.diagonal().array() += reg_;
    }
    Hff_inv_llt_.compute(Hff);
    const Eigen::ComputationInfo& info = Hff_inv_llt_.info();
    if (info != Eigen::Success) {
      throw_pretty("backward_error");
    }
    solution_.Hff_inv.setIdentity(nf_, nf_);
    Hff_inv_llt_.solveInPlace(solution_.Hff_inv);
    qf.noalias() += Hff * xf;
    if (nc_ != 0) {
      qf.noalias() += Hfc * xc;
    }
    dxf = -qf;
    Hff_inv_llt_.solveInPlace(dxf);
    dx_.setZero();
    for (std::size_t i = 0; i < nf_; ++i) {
      dx_(solution_.free_idx[i]) = dxf(i);
      g_(solution_.free_idx[i]) = -qf(i);
    }
    // Try different step lengths
    fold_ = Scalar(0.5) * x_.dot(H * x_) + q.dot(x_);
    if (reg_ != Scalar(0.)) {
      fold_ += Scalar(0.5) * reg_ * x_.squaredNorm();
    }
    for (typename std::vector<Scalar>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      Scalar steplength = *it;
      for (std::size_t i = 0; i < nx_; ++i) {
        xnew_(i) =
            std::max(std::min(x_(i) + steplength * dx_(i), ub(i)), lb(i));
      }
      fnew_ = Scalar(0.5) * xnew_.dot(H * xnew_) + q.dot(xnew_);
      if (reg_ != Scalar(0.)) {
        fnew_ += Scalar(0.5) * reg_ * xnew_.squaredNorm();
      }
      if (fold_ - fnew_ > th_acceptstep_ * g_.dot(x_ - xnew_)) {
        x_ = xnew_;
        break;
      }
    }
    // Check convergence
    if (qf.template lpNorm<Eigen::Infinity>() <= th_grad_) {
      solution_.x = x_;
      return solution_;
    }
  }
  solution_.x = x_;
  return solution_;
}

template <typename Scalar>
template <typename NewScalar>
BoxQPTpl<NewScalar> BoxQPTpl<Scalar>::cast() const {
  typedef BoxQPTpl<NewScalar> ReturnType;
  ReturnType ret(nx_, maxiter_, scalar_cast<NewScalar>(th_acceptstep_),
                 scalar_cast<NewScalar>(th_grad_),
                 scalar_cast<NewScalar>(reg_));
  return ret;
}

template <typename Scalar>
const BoxQPSolutionTpl<Scalar>& BoxQPTpl<Scalar>::get_solution() const {
  return solution_;
}

template <typename Scalar>
std::size_t BoxQPTpl<Scalar>::get_nx() const {
  return nx_;
}

template <typename Scalar>
std::size_t BoxQPTpl<Scalar>::get_maxiter() const {
  return maxiter_;
}

template <typename Scalar>
Scalar BoxQPTpl<Scalar>::get_th_acceptstep() const {
  return th_acceptstep_;
}

template <typename Scalar>
Scalar BoxQPTpl<Scalar>::get_th_grad() const {
  return th_grad_;
}

template <typename Scalar>
Scalar BoxQPTpl<Scalar>::get_reg() const {
  return reg_;
}

template <typename Scalar>
const std::vector<Scalar>& BoxQPTpl<Scalar>::get_alphas() const {
  return alphas_;
}

template <typename Scalar>
void BoxQPTpl<Scalar>::set_nx(const std::size_t nx) {
  nx_ = nx;
  x_.conservativeResize(nx);
  xnew_.conservativeResize(nx);
  g_.conservativeResize(nx);
  dx_.conservativeResize(nx);
  xo_.conservativeResize(nx);
  dxo_.conservativeResize(nx);
  qo_.conservativeResize(nx);
  Ho_.conservativeResize(nx, nx);
}

template <typename Scalar>
void BoxQPTpl<Scalar>::set_maxiter(const std::size_t maxiter) {
  maxiter_ = maxiter;
}

template <typename Scalar>
void BoxQPTpl<Scalar>::set_th_acceptstep(const Scalar th_acceptstep) {
  if (Scalar(0.) >= th_acceptstep && th_acceptstep >= Scalar(0.5)) {
    throw_pretty(
        "Invalid argument: " << "th_acceptstep value should between 0 and 0.5");
  }
  th_acceptstep_ = th_acceptstep;
}

template <typename Scalar>
void BoxQPTpl<Scalar>::set_th_grad(const Scalar th_grad) {
  if (Scalar(0.) > th_grad) {
    throw_pretty("Invalid argument: " << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

template <typename Scalar>
void BoxQPTpl<Scalar>::set_reg(const Scalar reg) {
  if (Scalar(0.) > reg) {
    throw_pretty("Invalid argument: " << "reg value has to be positive.");
  }
  reg_ = reg;
}

template <typename Scalar>
void BoxQPTpl<Scalar>::set_alphas(const std::vector<Scalar>& alphas) {
  Scalar prev_alpha = alphas[0];
  if (prev_alpha != Scalar(1.)) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    Scalar alpha = alphas[i];
    if (Scalar(0.) >= alpha) {
      throw_pretty("Invalid argument: " << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty(
          "Invalid argument: " << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
}

}  // namespace crocoddyl
