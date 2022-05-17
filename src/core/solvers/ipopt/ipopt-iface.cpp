#include <cassert>
#include <cmath>
#include <iostream>

#include "crocoddyl/core/solvers/ipopt/ipopt-iface.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

IpoptInterface::IpoptInterface(const boost::shared_ptr<crocoddyl::ShootingProblem> &problem)
    : problem_(problem),
      state_(problem_->get_runningModels()[0]->get_state()),
      nx_(problem_->get_nx()),
      ndx_(problem_->get_ndx()),
      nu_(problem_->get_nu_max()),
      T_(problem_->get_T()),
      nconst_(T_ * ndx_ + nx_),        // T*nx eq. constraints for dynamics , nx eq constraints for initial conditions
      nvar_(T_ * (ndx_ + nu_) + ndx_)  // Multiple shooting, states and controls
{
  xs_.resize(T_ + 1);
  us_.resize(T_);

  for (size_t i = 0; i < T_; i++) {
    xs_[i] = state_->zero();
    us_[i] = Eigen::VectorXd::Zero(nu_);
  }
  xs_[T_] = xs_[0];

  data_.x = state_->zero();
  data_.xnext = state_->zero();
  data_.dx = Eigen::VectorXd::Zero(ndx_);
  data_.dxnext = Eigen::VectorXd::Zero(ndx_);
  data_.x_diff = Eigen::VectorXd::Zero(ndx_);

  data_.control = Eigen::VectorXd::Zero(nu_);

  data_.Jsum_x = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Jsum_dx = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Jsum_xnext = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Jsum_dxnext = Eigen::MatrixXd::Zero(ndx_, ndx_);

  data_.Jdiff_xnext = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Jdiff_x = Eigen::MatrixXd::Zero(ndx_, ndx_);

  data_.Jg_dx = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Jg_dxnext = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Jg_u = Eigen::MatrixXd::Zero(ndx_, ndx_);

  data_.Ldx = Eigen::VectorXd::Zero(ndx_);
  data_.Ldxdx = Eigen::MatrixXd::Zero(ndx_, ndx_);
  data_.Ldxu = Eigen::MatrixXd::Zero(ndx_, nu_);
}

IpoptInterface::~IpoptInterface() {}

bool IpoptInterface::get_nlp_info(Ipopt::Index &n, Ipopt::Index &m, Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                                  IndexStyleEnum &index_style) {
  n = nvar_;

  m = nconst_;

  // Jacobian nonzeros for dynamic constraints
  nnz_jac_g = T_ * ndx_ * (2 * ndx_ + nu_);

  // Jacobian nonzeros for initial condition
  nnz_jac_g += ndx_;

  // Hessian is only affected by costs
  // Running Costs
  std::size_t nonzero = 0;
  for (size_t i = 1; i <= (ndx_ + nu_); i++) {
    nonzero += i;
  }
  nnz_h_lag = T_ * nonzero;

  // Terminal Costs
  nonzero = 0;
  for (size_t i = 1; i <= ndx_; i++) {
    nonzero += i;
  }
  nnz_h_lag += nonzero;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}
// [TNLP_get_nlp_info]

// [TNLP_get_bounds_info]
// returns the variable bounds
bool IpoptInterface::get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l, Ipopt::Number *x_u, Ipopt::Index m,
                                     Ipopt::Number *g_l, Ipopt::Number *g_u) {
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.
  assert(n == nvar_);
  assert(m == nconst_);

  // // the variables have lower bounds of 1
  // for (Ipopt::Index i = 0; i < n; i++) {
  //     x_l[i] = -2e19;
  //     x_u[i] = 2e19;
  // }

  // Adding bounds
  for (size_t i = 0; i < T_; i++) {
    // Running state bounds
    for (size_t j = 0; j < ndx_; j++) {
      x_l[i * (ndx_ + nu_) + j] = -2e19;
      x_u[i * (ndx_ + nu_) + j] = 2e19;
    }
    // Control bounds
    for (size_t j = 0; j < nu_; j++) {
      x_l[i * (ndx_ + nu_) + ndx_ + j] = problem_->get_runningModels()[i]->get_has_control_limits()
                                             ? problem_->get_runningModels()[i]->get_u_lb()(j)
                                             : -2e19;
      x_u[i * (ndx_ + nu_) + ndx_ + j] = problem_->get_runningModels()[i]->get_has_control_limits()
                                             ? problem_->get_runningModels()[i]->get_u_ub()(j)
                                             : 2e19;
    }
  }
  // Final state bounds
  for (size_t j = 0; j < ndx_; j++) {
    x_l[T_ * (ndx_ + nu_) + j] = -2e19;
    x_u[T_ * (ndx_ + nu_) + j] = 2e19;
  }

  // Dynamics
  for (Ipopt::Index i = 0; i < nconst_ - nx_; i++) {
    g_l[i] = 0;
    g_u[i] = 0;
  }

  // Initital conditions
  for (Ipopt::Index i = 0; i < nx_; i++) {
    g_l[nconst_ - nx_ + i] = problem_->get_x0()[i];
    g_u[nconst_ - nx_ + i] = problem_->get_x0()[i];
  }

  return true;
}
// [TNLP_get_bounds_info]

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool IpoptInterface::get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x, bool init_z, Ipopt::Number *z_L,
                                        Ipopt::Number *z_U, Ipopt::Index m, bool init_lambda, Ipopt::Number *lambda) {
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // initialize to the given starting point
  // State variable are always at 0 since they represent increments from the given initial point
  for (size_t i = 0; i < T_; i++) {
    for (size_t j = 0; j < ndx_; j++) {
      x[i * (ndx_ + nu_) + j] = 0;
    }

    for (size_t j = 0; j < nu_; j++) {
      x[i * (ndx_ + nu_) + ndx_ + j] = us_[i](j);
    }
  }

  for (size_t j = 0; j < ndx_; j++) {
    x[T_ * (ndx_ + nu_) + j] = 0;
  }

  return true;
}
// [TNLP_get_starting_point]

// [TNLP_eval_f]
// returns the value of the objective function
bool IpoptInterface::eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value) {
  assert(n == nvar_);

  // Running costs
  for (size_t i = 0; i < T_; i++) {
    data_.dx = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
    data_.control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

    state_->integrate(xs_[i], data_.dx, data_.x);

    problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], data_.x, data_.control);

    obj_value += problem_->get_runningDatas()[i]->cost;
  }

  // Terminal costs
  data_.dx = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);

  state_->integrate(xs_[T_], data_.dx, data_.x);

  problem_->get_terminalModel()->calc(problem_->get_terminalData(), data_.x);
  obj_value += problem_->get_terminalData()->cost;

  return true;
}
// [TNLP_eval_f]

// [TNLP_eval_grad_f]
// return the gradient of the objective function grad_{x} f(x)
bool IpoptInterface::eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f) {
  assert(n == nvar_);

  for (size_t i = 0; i < T_; i++) {
    data_.dx = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
    data_.control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

    state_->integrate(xs_[i], data_.dx, data_.x);

    state_->Jintegrate(xs_[i], data_.dx, data_.Jsum_x, data_.Jsum_dx, crocoddyl::second, crocoddyl::setto);
    problem_->get_runningModels()[i]->calcDiff(problem_->get_runningDatas()[i], data_.x, data_.control);
    data_.Ldx = data_.Jsum_dx.transpose() * problem_->get_runningDatas()[i]->Lx;

    for (size_t j = 0; j < ndx_; j++) {
      grad_f[i * (ndx_ + nu_) + j] = data_.Ldx(j);
    }

    for (size_t j = 0; j < nu_; j++) {
      grad_f[i * (ndx_ + nu_) + ndx_ + j] = problem_->get_runningDatas()[i]->Lu(j);
    }
  }

  data_.dx = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);

  state_->integrate(xs_[T_], data_.dx, data_.x);
  state_->Jintegrate(xs_[T_], data_.dx, data_.Jsum_x, data_.Jsum_dx, crocoddyl::second, crocoddyl::setto);

  problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), data_.x);
  data_.Ldx = data_.Jsum_dx.transpose() * problem_->get_terminalData()->Lx;

  for (size_t j = 0; j < ndx_; j++) {
    grad_f[T_ * (ndx_ + nu_) + j] = data_.Ldx(j);
  }

  return true;
}
// [TNLP_eval_grad_f]

// [TNLP_eval_g]
// return the value of the constraints: g(x)
bool IpoptInterface::eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Number *g) {
  assert(n == nvar_);
  assert(m == nconst_);

  // Dynamic constraints
  for (size_t i = 0; i < T_; i++) {
    data_.dx = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
    data_.dxnext = Eigen::VectorXd::Map(x + (i + 1) * (ndx_ + nu_), ndx_);
    data_.control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

    state_->integrate(xs_[i], data_.dx, data_.x);
    state_->integrate(xs_[i + 1], data_.dxnext, data_.xnext);

    problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], data_.x, data_.control);

    state_->diff(problem_->get_runningDatas()[i]->xnext, data_.xnext, data_.x_diff);

    for (size_t j = 0; j < ndx_; j++) {
      g[i * ndx_ + j] = data_.x_diff[j];
    }
  }

  // Initial conditions
  data_.dx = Eigen::VectorXd::Map(x, ndx_);
  state_->integrate(xs_[0], data_.dx, data_.x);

  for (size_t j = 0; j < nx_; j++) {
    g[T_ * ndx_ + j] = data_.x[j];
  }

  return true;
}

// [TNLP_eval_g]

// [TNLP_eval_jac_g]
// return the structure or values of the Jacobian
bool IpoptInterface::eval_jac_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m,
                                Ipopt::Index nele_jac, Ipopt::Index *iRow, Ipopt::Index *jCol, Ipopt::Number *values) {
  assert(n == nvar_);
  assert(m == nconst_);

  if (values == NULL) {
    std::size_t idx_value = 0;
    // Dynamic constraints
    for (size_t idx_block = 0; idx_block < T_; idx_block++) {
      for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
        for (size_t idx_col = 0; idx_col < (2 * ndx_ + nu_); idx_col++) {
          iRow[idx_value] = idx_block * ndx_ + idx_row;
          jCol[idx_value] = idx_block * (ndx_ + nu_) + idx_col;
          idx_value++;
        }
      }
    }

    // Initial condition
    for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
      for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
        if (idx_row == idx_col) {
          iRow[idx_value] = T_ * ndx_ + idx_row;
          jCol[idx_value] = idx_col;
          idx_value++;
        }
      }
    }

    assert(nele_jac == idx_value);

  } else {
    std::size_t idx_value = 0;

    // Dynamic constraints
    for (size_t idx_block = 0; idx_block < T_; idx_block++) {
      data_.dx = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_), ndx_);
      data_.dxnext = Eigen::VectorXd::Map(x + (idx_block + 1) * (ndx_ + nu_), ndx_);
      data_.control = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_) + ndx_, nu_);

      state_->integrate(xs_[idx_block], data_.dx, data_.x);
      state_->integrate(xs_[idx_block + 1], data_.dxnext, data_.xnext);

      problem_->get_runningModels()[idx_block]->calc(problem_->get_runningDatas()[idx_block], data_.x, data_.control);
      problem_->get_runningModels()[idx_block]->calcDiff(problem_->get_runningDatas()[idx_block], data_.x,
                                                         data_.control);

      state_->Jintegrate(xs_[idx_block], data_.dx, data_.Jsum_x, data_.Jsum_dx, crocoddyl::second, crocoddyl::setto);
      state_->Jintegrate(xs_[idx_block + 1], data_.dxnext, data_.Jsum_xnext, data_.Jsum_dxnext, crocoddyl::second,
                         crocoddyl::setto);
      state_->Jdiff(problem_->get_runningDatas()[idx_block]->xnext, data_.xnext, data_.Jdiff_x, data_.Jdiff_xnext,
                    crocoddyl::both);

      data_.Jg_dx = data_.Jdiff_x * problem_->get_runningDatas()[idx_block]->Fx * data_.Jsum_dx;
      data_.Jg_u = data_.Jdiff_x * problem_->get_runningDatas()[idx_block]->Fu;
      data_.Jg_dxnext = data_.Jdiff_xnext * data_.Jsum_dxnext;

      for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
        for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
          values[idx_value] = data_.Jg_dx(idx_row, idx_col);
          idx_value++;
        }

        for (size_t idx_col = 0; idx_col < nu_; idx_col++) {
          values[idx_value] = data_.Jg_u(idx_row, idx_col);
          idx_value++;
        }

        // This could be more optimized since there are a lot of zeros!
        for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
          values[idx_value] = data_.Jg_dxnext(idx_row, idx_col);
          idx_value++;
        }
      }
    }

    // Initial condition
    data_.dx = Eigen::VectorXd::Map(x, ndx_);

    state_->Jintegrate(xs_[0], data_.dx, data_.Jsum_x, data_.Jsum_dx, crocoddyl::second, crocoddyl::setto);

    for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
      for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
        if (idx_row == idx_col) {
          values[idx_value] = data_.Jsum_dx(idx_row, idx_col);
          idx_value++;
        }
      }
    }
  }

  return true;
}
// [TNLP_eval_jac_g]

// [TNLP_eval_h]
// return the structure or values of the Hessian
bool IpoptInterface::eval_h(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number obj_factor,
                            Ipopt::Index m, const Ipopt::Number *lambda, bool new_lambda, Ipopt::Index nele_hess,
                            Ipopt::Index *iRow, Ipopt::Index *jCol, Ipopt::Number *values) {
  assert(n == nvar_);
  assert(m == nconst_);

  if (values == NULL) {
    // return the structure. This is a symmetric matrix, fill the lower left
    // triangle only

    // Running Costs
    std::size_t idx_value = 0;
    for (size_t idx_block = 0; idx_block < T_; idx_block++) {
      for (size_t idx_row = 0; idx_row < ndx_ + nu_; idx_row++) {
        for (size_t idx_col = 0; idx_col < ndx_ + nu_; idx_col++) {
          // We need the lower triangular matrix
          if (idx_col > idx_row) {
            break;
          }
          iRow[idx_value] = idx_block * (ndx_ + nu_) + idx_row;
          jCol[idx_value] = idx_block * (ndx_ + nu_) + idx_col;
          idx_value++;
        }
      }
    }

    // Terminal costs
    for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
      for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
        // We need the lower triangular matrix
        if (idx_col > idx_row) {
          break;
        }
        iRow[idx_value] = T_ * (ndx_ + nu_) + idx_row;
        jCol[idx_value] = T_ * (ndx_ + nu_) + idx_col;
        idx_value++;
      }
    }

    assert(idx_value == nele_hess);
  } else {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only
    std::size_t idx_value = 0;

    // Running Costs
    for (size_t idx_block = 0; idx_block < T_; idx_block++) {
      data_.dx = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_), ndx_);
      data_.control = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_) + ndx_, nu_);

      state_->integrate(xs_[idx_block], data_.dx, data_.x);

      problem_->get_runningModels()[idx_block]->calcDiff(problem_->get_runningDatas()[idx_block], data_.x,
                                                         data_.control);

      state_->Jintegrate(xs_[idx_block], data_.dx, data_.Jsum_x, data_.Jsum_dx, crocoddyl::second, crocoddyl::setto);
      data_.Ldxdx = data_.Jsum_dx.transpose() * problem_->get_runningDatas()[idx_block]->Lxx * data_.Jsum_dx;
      data_.Ldxu = data_.Jsum_dx.transpose() * problem_->get_runningDatas()[idx_block]->Lxu;

      for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
        for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
          // We need the lower triangular matrix
          if (idx_col > idx_row) {
            break;
          }
          values[idx_value] = obj_factor * data_.Ldxdx(idx_row, idx_col);
          idx_value++;
        }
      }

      for (size_t idx_row = 0; idx_row < nu_; idx_row++) {
        for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
          values[idx_value] = obj_factor * data_.Ldxu(idx_col, idx_row);
          idx_value++;
        }

        for (size_t idx_col = 0; idx_col < nu_; idx_col++) {
          if (idx_col > idx_row) {
            break;
          }
          values[idx_value] = obj_factor * problem_->get_runningDatas()[idx_block]->Luu(idx_row, idx_col);
          idx_value++;
        }
      }
    }

    // Terminal costs
    data_.dx = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);

    state_->integrate(xs_[T_], data_.dx, data_.x);
    problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), data_.x);
    state_->Jintegrate(xs_[T_], data_.dx, data_.Jsum_x, data_.Jsum_dx, crocoddyl::second, crocoddyl::setto);

    data_.Ldxdx = data_.Jsum_dx.transpose() * problem_->get_terminalData()->Lxx * data_.Jsum_dx;

    for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
      for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
        // We need the lower triangular matrix
        if (idx_col > idx_row) {
          break;
        }
        values[idx_value] = data_.Ldxdx(idx_row, idx_col);
        idx_value++;
      }
    }
  }

  return true;
}
// [TNLP_eval_h]

// [TNLP_finalize_solution]
void IpoptInterface::finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number *x,
                                       const Ipopt::Number *z_L, const Ipopt::Number *z_U, Ipopt::Index m,
                                       const Ipopt::Number *g, const Ipopt::Number *lambda, Ipopt::Number obj_value,
                                       const Ipopt::IpoptData *ip_data, Ipopt::IpoptCalculatedQuantities *ip_cq) {
  // Copy the solution to vector once solver is finished
  for (size_t i = 0; i < T_; i++) {
    data_.dx = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
    data_.control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

    state_->integrate(xs_[i], data_.dx, data_.x);
    xs_[i] = data_.x;
    us_[i] = data_.control;
  }

  data_.dx = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);
  state_->integrate(xs_[T_], data_.dx, data_.x);

  xs_[T_] = data_.x;
}
// [TNLP_finalize_solution]

// [TNLP_intermediate_callback]
bool IpoptInterface::intermediate_callback(Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value,
                                           Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu,
                                           Ipopt::Number d_norm, Ipopt::Number regularization_size,
                                           Ipopt::Number alpha_du, Ipopt::Number alpha_pr, Ipopt::Index ls_trials,
                                           const Ipopt::IpoptData *ip_data, Ipopt::IpoptCalculatedQuantities *ip_cq) {
  return true;
}

void IpoptInterface::set_xs(const std::vector<Eigen::VectorXd> &xs) { xs_ = xs; }

void IpoptInterface::set_us(const std::vector<Eigen::VectorXd> &us) { us_ = us; }

const std::size_t &IpoptInterface::get_nvar() const { return nvar_; }

const std::vector<Eigen::VectorXd> &IpoptInterface::get_xs() const { return xs_; }

const std::vector<Eigen::VectorXd> &IpoptInterface::get_us() const { return us_; }

const boost::shared_ptr<crocoddyl::ShootingProblem> &IpoptInterface::get_problem() const { return problem_; }
