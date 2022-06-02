///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_IPOPT

#include <cmath>

#include <iostream>

#include "crocoddyl/core/solvers/ipopt/ipopt-iface.hpp"

namespace crocoddyl {

IpoptInterface::IpoptInterface(const boost::shared_ptr<ShootingProblem> &problem)
    : problem_(problem), T_(problem_->get_T()) {
  xs_.resize(T_ + 1);
  us_.resize(T_);
  datas_.resize(T_ + 1);

  nconst_ = 0;
  nvar_ = 0;

  const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T_; ++t) {
    const std::size_t nxi = models[t]->get_state()->get_nx();
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    xs_[t] = models[t]->get_state()->zero();
    us_[t] = Eigen::VectorXd::Zero(nui);

    nconst_ += ndxi;      // T*ndx eq. constraints for dynamics
    nvar_ += ndxi + nui;  // Multiple shooting, states and controls

    datas_[t] = createData(nxi, ndxi, nui);
  }

  // Initial condition
  nconst_ += models[0]->get_state()->get_ndx();

  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
  const std::size_t nxi = model->get_state()->get_nx();
  const std::size_t ndxi = model->get_state()->get_ndx();

  nvar_ += ndxi;  // final node

  xs_[T_] = model->get_state()->zero();
  datas_[T_] = createData(nxi, ndxi, 0);
}

IpoptInterface::~IpoptInterface() {}

bool IpoptInterface::get_nlp_info(Ipopt::Index &n, Ipopt::Index &m, Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                                  IndexStyleEnum &index_style) {
  n = nvar_;  // number of variables

  m = nconst_;  // number of constraints

  nnz_jac_g = 0;  // Jacobian nonzeros for dynamic constraints
  nnz_h_lag = 0;  // Hessian nonzeros (only lower triangular part)

  const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T_; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t ndxi_next =
        t + 1 == T_ ? problem_->get_terminalModel()->get_state()->get_ndx() : models[t + 1]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    nnz_jac_g += ndxi * (ndxi + ndxi_next + nui);

    // Hessian
    std::size_t nonzero = 0;
    for (std::size_t i = 1; i <= (ndxi + nui); i++) {
      nonzero += i;
    }
    nnz_h_lag += nonzero;
  }

  // Initial condition
  nnz_jac_g += models[0]->get_state()->get_ndx() * models[0]->get_state()->get_ndx();

  // Hessian nonzero for the terminal cost
  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
  const std::size_t ndxi = model->get_state()->get_ndx();

  std::size_t nonzero = 0;
  for (std::size_t i = 1; i <= ndxi; i++) {
    nonzero += i;
  }
  nnz_h_lag += nonzero;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

bool IpoptInterface::get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l, Ipopt::Number *x_u, Ipopt::Index m,
                                     Ipopt::Number *g_l, Ipopt::Number *g_u) {
  assert_pretty(n == nvar_, "Inconsistent number of decision variables");
  assert_pretty(m == nconst_, "Inconsistent number of constraints");

  // Adding bounds
  const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
  std::size_t ixu = 0;
  for (std::size_t t = 0; t < T_; ++t) {
    // Running state bounds
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    for (std::size_t j = 0; j < ndxi; j++) {
      x_l[ixu + j] = std::numeric_limits<double>::lowest();
      x_u[ixu + j] = std::numeric_limits<double>::max();
    }

    // Control bounds
    for (std::size_t j = 0; j < nui; j++) {
      x_l[ixu + ndxi + j] =
          models[t]->get_has_control_limits() ? models[t]->get_u_lb()(j) : std::numeric_limits<double>::lowest();
      x_u[ixu + ndxi + j] =
          models[t]->get_has_control_limits() ? models[t]->get_u_ub()(j) : std::numeric_limits<double>::max();
    }

    ixu += ndxi + nui;
  }

  {
    const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
    const std::size_t ndxi = model->get_state()->get_ndx();

    // Final state bounds
    for (std::size_t j = 0; j < ndxi; j++) {
      x_l[ixu + j] = std::numeric_limits<double>::lowest();
      x_u[ixu + j] = std::numeric_limits<double>::max();
    }
  }

  // Dynamics & Initial conditions (all equal to zero)
  for (Ipopt::Index i = 0; i < nconst_; i++) {
    g_l[i] = 0;
    g_u[i] = 0;
  }

  return true;
}

bool IpoptInterface::get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x, bool init_z, Ipopt::Number *z_L,
                                        Ipopt::Number *z_U, Ipopt::Index m, bool init_lambda, Ipopt::Number *lambda) {
  assert_pretty(init_x == true, "Make sure to provide initial value for primal variables");
  assert_pretty(init_z == false, "Cannot provide initial value for bound multipliers");
  assert_pretty(init_lambda == false, "Cannot provide initial value for constraint multipliers");

  // initialize to the given starting point
  // State variables are always at 0 since they represent increments from the given initial point
  const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
  std::size_t ixu = 0;
  for (std::size_t t = 0; t < T_; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    for (std::size_t j = 0; j < ndxi; j++) {
      x[ixu + j] = 0;
    }

    for (std::size_t j = 0; j < nui; j++) {
      x[ixu + ndxi + j] = us_[t](j);
    }

    ixu += ndxi + nui;
  }

  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
  const std::size_t ndxi = model->get_state()->get_ndx();
  for (std::size_t j = 0; j < ndxi; j++) {
    x[ixu + j] = 0;
  }

  return true;
}

bool IpoptInterface::eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value) {
  assert_pretty(n == nvar_, "Inconsistent number of decision variables");

  // Running costs
  std::size_t ixu = 0;
  for (std::size_t t = 0; t < T_; ++t) {
    const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[t];
    const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();

    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu + ndxi, nui);

    model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);

    model->calc(data, datas_[t]->x, datas_[t]->u);

    obj_value += data->cost;

    ixu += ndxi + nui;
  }

  // Terminal costs
  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
  const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_terminalData();
  const std::size_t ndxi = model->get_state()->get_ndx();

  datas_[T_]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
  model->get_state()->integrate(xs_[T_], datas_[T_]->dx, datas_[T_]->x);

  model->calc(data, datas_[T_]->x);
  obj_value += data->cost;

  return true;
}

bool IpoptInterface::eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f) {
  assert_pretty(n == nvar_, "Inconsistent number of decision variables");

  std::size_t ixu = 0;

  for (std::size_t t = 0; t < T_; ++t) {
    const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[t];
    const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();

    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu + ndxi, nui);

    model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    model->get_state()->Jintegrate(xs_[t], datas_[t]->dx, datas_[t]->Jint_dx, datas_[t]->Jint_dx, second, setto);

    model->calc(data, datas_[t]->x, datas_[t]->u);
    model->calcDiff(data, datas_[t]->x, datas_[t]->u);
    datas_[t]->Ldx.noalias() = datas_[t]->Jint_dx.transpose() * data->Lx;

    for (std::size_t j = 0; j < ndxi; j++) {
      grad_f[ixu + j] = datas_[t]->Ldx(j);
    }

    for (std::size_t j = 0; j < nui; j++) {
      grad_f[ixu + ndxi + j] = data->Lu(j);
    }
    ixu += ndxi + nui;
  }

  // Terminal model
  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
  const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_terminalData();
  const std::size_t ndxi = model->get_state()->get_ndx();

  datas_[T_]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);

  model->get_state()->integrate(xs_[T_], datas_[T_]->dx, datas_[T_]->x);
  model->get_state()->Jintegrate(xs_[T_], datas_[T_]->dx, datas_[T_]->Jint_dx, datas_[T_]->Jint_dx, second, setto);

  model->calc(data, datas_[T_]->x);
  model->calcDiff(data, datas_[T_]->x);
  datas_[T_]->Ldx.noalias() = datas_[T_]->Jint_dx.transpose() * data->Lx;

  for (std::size_t j = 0; j < ndxi; j++) {
    grad_f[ixu + j] = datas_[T_]->Ldx(j);
  }

  return true;
}

bool IpoptInterface::eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Number *g) {
  assert_pretty(n == nvar_, "Inconsistent number of decision variables");
  assert_pretty(m == nconst_, "Inconsistent number of constraints");

  std::size_t ixu = 0;
  std::size_t ix = 0;
  // Dynamic constraints
  for (std::size_t t = 0; t < T_; ++t) {
    const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[t];
    const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();

    const boost::shared_ptr<ActionModelAbstract> &model_next =
        t + 1 == T_ ? problem_->get_terminalModel() : problem_->get_runningModels()[t + 1];
    const boost::shared_ptr<ActionDataAbstract> &data_next =
        t + 1 == T_ ? problem_->get_terminalData() : problem_->get_runningDatas()[t + 1];
    const std::size_t ndxi_next = model_next->get_state()->get_ndx();

    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu + ndxi, nui);
    datas_[t + 1]->dx = Eigen::VectorXd::Map(x + ixu + ndxi + nui, ndxi_next);

    model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    model_next->get_state()->integrate(xs_[t + 1], datas_[t + 1]->dx, datas_[t + 1]->x);

    model->calc(data, datas_[t]->x, datas_[t]->u);

    // This computes: datas_[t+1]->x - data->xnext (x_next - f(x, u))
    model->get_state()->diff(data->xnext, datas_[t + 1]->x, datas_[t]->x_diff);

    for (std::size_t j = 0; j < ndxi; j++) {
      g[ix + j] = datas_[t]->x_diff[j];
    }

    ixu += ndxi + nui;
    ix += ndxi;
  }

  // Initial conditions
  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[0];
  const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[0];
  const std::size_t ndxi = model->get_state()->get_ndx();
  datas_[0]->dx = Eigen::VectorXd::Map(x, ndxi);
  model->get_state()->integrate(xs_[0], datas_[0]->dx, datas_[0]->x);
  // x(0) - x_0
  model->get_state()->diff(datas_[0]->x, problem_->get_x0(), datas_[0]->x_diff);

  for (std::size_t j = 0; j < ndxi; j++) {
    g[ix + j] = datas_[0]->x_diff[j];
  }

  return true;
}

bool IpoptInterface::eval_jac_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m,
                                Ipopt::Index nele_jac, Ipopt::Index *iRow, Ipopt::Index *jCol, Ipopt::Number *values) {
  assert_pretty(n == nvar_, "Inconsistent number of decision variables");
  assert_pretty(m == nconst_, "Inconsistent number of constraints");

  if (values == NULL) {
    // Dynamic constraints
    const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
    std::size_t idx_value = 0;
    std::size_t ixu = 0;
    std::size_t ix = 0;
    for (std::size_t t = 0; t < T_; ++t) {
      const std::size_t ndxi = models[t]->get_state()->get_ndx();
      const std::size_t nui = models[t]->get_nu();
      const std::size_t ndxi_next =
          t + 1 == T_ ? problem_->get_terminalModel()->get_state()->get_ndx() : models[t + 1]->get_state()->get_ndx();
      for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
        for (std::size_t idx_col = 0; idx_col < (ndxi + nui + ndxi_next); idx_col++) {
          iRow[idx_value] = ix + idx_row;
          jCol[idx_value] = ixu + idx_col;
          idx_value++;
        }
      }
      ixu += ndxi + nui;
      ix += ndxi;
    }

    // Initial condition
    const std::size_t ndxi = models[0]->get_state()->get_ndx();
    for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
      for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
        iRow[idx_value] = ix + idx_row;
        jCol[idx_value] = idx_col;
        idx_value++;
      }
    }

    assert_pretty(nele_jac == idx_value,
                  "Number of jacobian elements set does not coincide with the total non-zero Jacobian values");

  } else {
    std::size_t idx_value = 0;
    std::size_t ixu = 0;
    std::size_t ix = 0;
    // Dynamic constraints
    for (std::size_t t = 0; t < T_; ++t) {
      const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[t];
      const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[t];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();

      const boost::shared_ptr<ActionModelAbstract> &model_next =
          t + 1 == T_ ? problem_->get_terminalModel() : problem_->get_runningModels()[t + 1];
      const boost::shared_ptr<ActionDataAbstract> &data_next =
          t + 1 == T_ ? problem_->get_terminalData() : problem_->get_runningDatas()[t + 1];
      const std::size_t ndxi_next = model_next->get_state()->get_ndx();

      datas_[t]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
      datas_[t]->u = Eigen::VectorXd::Map(x + ixu + ndxi, nui);
      datas_[t + 1]->dx = Eigen::VectorXd::Map(x + ixu + ndxi + nui, ndxi_next);

      model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
      model_next->get_state()->integrate(xs_[t + 1], datas_[t + 1]->dx, datas_[t + 1]->x);

      model->calcDiff(data, datas_[t]->x, datas_[t]->u);

      model_next->get_state()->Jintegrate(xs_[t + 1], datas_[t + 1]->dx, datas_[t + 1]->Jint_dx,
                                          datas_[t + 1]->Jint_dx, second, setto);  // datas_[t]->Jsum_dxnext == eq. 81
      model->get_state()->Jdiff(data->xnext, datas_[t + 1]->x, datas_[t]->Jdiff_x, datas_[t + 1]->Jdiff_x,
                                both);  // datas_[t+1]->Jdiff_x == eq. 83, datas_[t]->Jdiff_x == eq.82
      model->get_state()->Jintegrate(xs_[t], datas_[t]->dx, datas_[t]->Jint_dx, datas_[t]->Jint_dx, second,
                                     setto);  // datas_[t]->Jsum_dx == eq. 81

      datas_[t + 1]->Jg_dx.noalias() = datas_[t + 1]->Jdiff_x * datas_[t + 1]->Jint_dx;  // chain rule
      datas_[t]->Jg_dx.noalias() = datas_[t]->Jdiff_x * data->Fx * datas_[t]->Jint_dx;
      datas_[t]->Jg_u.noalias() = datas_[t]->Jdiff_x * data->Fu;

      for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
        for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
          values[idx_value] = datas_[t]->Jg_dx(idx_row, idx_col);
          idx_value++;
        }

        for (std::size_t idx_col = 0; idx_col < nui; idx_col++) {
          values[idx_value] = datas_[t]->Jg_u(idx_row, idx_col);
          idx_value++;
        }

        // This could be more optimized since there are a lot of zeros!
        for (std::size_t idx_col = 0; idx_col < ndxi_next; idx_col++) {
          values[idx_value] = datas_[t + 1]->Jg_dx(idx_row, idx_col);
          idx_value++;
        }
      }
      ixu += ndxi + nui;
      ix += ndxi;
    }

    // Initial condition
    const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[0];
    const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[0];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();

    datas_[0]->dx = Eigen::VectorXd::Map(x, ndxi);
    model->get_state()->integrate(xs_[0], datas_[0]->dx, datas_[0]->x);
    model->get_state()->Jdiff(datas_[0]->x, problem_->get_x0(), datas_[0]->Jdiff_x, datas_[0]->Jdiff_x, first);
    model->get_state()->Jintegrate(xs_[0], datas_[0]->dx, datas_[0]->Jint_dx, datas_[0]->Jint_dx, second, setto);

    datas_[0]->Jg_ic.noalias() = datas_[0]->Jdiff_x * datas_[0]->Jint_dx;

    for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
      for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
        values[idx_value] = datas_[0]->Jg_ic(idx_row, idx_col);
        idx_value++;
      }
    }
  }

  return true;
}

bool IpoptInterface::eval_h(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number obj_factor,
                            Ipopt::Index m, const Ipopt::Number *lambda, bool new_lambda, Ipopt::Index nele_hess,
                            Ipopt::Index *iRow, Ipopt::Index *jCol, Ipopt::Number *values) {
  assert_pretty(n == nvar_, "Inconsistent number of decision variables");
  assert_pretty(m == nconst_, "Inconsistent number of constraints");

  if (values == NULL) {
    // return the structure. This is a symmetric matrix, fill the lower left
    // triangle only

    // Running Costs
    const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
    std::size_t idx_value = 0;
    std::size_t ixu = 0;
    for (std::size_t t = 0; t < T_; ++t) {
      const std::size_t ndxi = models[t]->get_state()->get_ndx();
      const std::size_t nui = models[t]->get_nu();
      for (std::size_t idx_row = 0; idx_row < ndxi + nui; idx_row++) {
        for (std::size_t idx_col = 0; idx_col < ndxi + nui; idx_col++) {
          // We need the lower triangular matrix
          if (idx_col > idx_row) {
            break;
          }
          iRow[idx_value] = ixu + idx_row;
          jCol[idx_value] = ixu + idx_col;
          idx_value++;
        }
      }
      ixu += ndxi + nui;
    }

    // Terminal costs
    const std::size_t ndxi = problem_->get_terminalModel()->get_state()->get_ndx();
    for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
      for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
        // We need the lower triangular matrix
        if (idx_col > idx_row) {
          break;
        }
        iRow[idx_value] = ixu + idx_row;
        jCol[idx_value] = ixu + idx_col;
        idx_value++;
      }
    }

    assert_pretty(idx_value == nele_hess,
                  "Number of Hessian elements set does not coincide with the total non-zero Hessian values");
  } else {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only
    std::size_t idx_value = 0;
    std::size_t ixu = 0;
    // Running Costs
    for (std::size_t t = 0; t < T_; ++t) {
      const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_runningModels()[t];
      const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_runningDatas()[t];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();

      datas_[t]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
      datas_[t]->u = Eigen::VectorXd::Map(x + ixu + ndxi, nui);

      model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);

      model->calcDiff(data, datas_[t]->x, datas_[t]->u);  // this might be removed

      model->get_state()->Jintegrate(xs_[t], datas_[t]->dx, datas_[t]->Jint_dx, datas_[t]->Jint_dx, second, setto);
      datas_[t]->Ldxdx = datas_[t]->Jint_dx.transpose() * problem_->get_runningDatas()[t]->Lxx * datas_[t]->Jint_dx;
      datas_[t]->Ldxu = datas_[t]->Jint_dx.transpose() * problem_->get_runningDatas()[t]->Lxu;

      for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
        for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
          // We need the lower triangular matrix
          if (idx_col > idx_row) {
            break;
          }
          values[idx_value] = obj_factor * datas_[t]->Ldxdx(idx_row, idx_col);
          idx_value++;
        }
      }

      for (std::size_t idx_row = 0; idx_row < nui; idx_row++) {
        for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
          values[idx_value] = obj_factor * datas_[t]->Ldxu(idx_col, idx_row);
          idx_value++;
        }

        for (std::size_t idx_col = 0; idx_col < nui; idx_col++) {
          if (idx_col > idx_row) {
            break;
          }
          values[idx_value] = obj_factor * data->Luu(idx_row, idx_col);
          idx_value++;
        }
      }
      ixu += ndxi + nui;
    }

    // Terminal costs
    const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
    const boost::shared_ptr<ActionDataAbstract> &data = problem_->get_terminalData();
    const std::size_t ndxi = model->get_state()->get_ndx();

    datas_[T_]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
    model->get_state()->integrate(xs_[T_], datas_[T_]->dx, datas_[T_]->x);
    model->calc(data, datas_[T_]->x);
    model->calcDiff(data, datas_[T_]->x);
    model->get_state()->Jintegrate(xs_[T_], datas_[T_]->dx, datas_[T_]->Jint_dx, datas_[T_]->Jint_dx, second, setto);

    datas_[T_]->Ldxdx.noalias() = datas_[T_]->Jint_dx.transpose() * data->Lxx * datas_[T_]->Jint_dx;

    for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
      for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
        // We need the lower triangular matrix
        if (idx_col > idx_row) {
          break;
        }
        values[idx_value] = datas_[T_]->Ldxdx(idx_row, idx_col);
        idx_value++;
      }
    }
  }

  return true;
}

void IpoptInterface::finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number *x,
                                       const Ipopt::Number *z_L, const Ipopt::Number *z_U, Ipopt::Index m,
                                       const Ipopt::Number *g, const Ipopt::Number *lambda, Ipopt::Number obj_value,
                                       const Ipopt::IpoptData *ip_data, Ipopt::IpoptCalculatedQuantities *ip_cq) {
  // Copy the solution to vector once solver is finished
  const std::vector<boost::shared_ptr<ActionModelAbstract> > &models = problem_->get_runningModels();
  std::size_t ixu = 0;
  for (std::size_t t = 0; t < T_; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu + ndxi, nui);

    models[t]->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    xs_[t] = datas_[t]->x;
    us_[t] = datas_[t]->u;

    ixu += ndxi + nui;
  }

  // Terminal node
  const boost::shared_ptr<ActionModelAbstract> &model = problem_->get_terminalModel();
  const std::size_t ndxi = model->get_state()->get_ndx();
  datas_[T_]->dx = Eigen::VectorXd::Map(x + ixu, ndxi);
  model->get_state()->integrate(xs_[T_], datas_[T_]->dx, datas_[T_]->x);

  xs_[T_] = datas_[T_]->x;
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

boost::shared_ptr<IpoptInterfaceData> IpoptInterface::createData(const std::size_t nx, const std::size_t ndx,
                                                                 const std::size_t nu) {
  return boost::allocate_shared<IpoptInterfaceData>(Eigen::aligned_allocator<IpoptInterfaceData>(), nx, ndx, nu);
}

void IpoptInterface::set_xs(const std::vector<Eigen::VectorXd> &xs) { xs_ = xs; }

void IpoptInterface::set_us(const std::vector<Eigen::VectorXd> &us) { us_ = us; }

std::size_t IpoptInterface::get_nvar() const { return nvar_; }

std::size_t IpoptInterface::get_nconst() const { return nconst_; }

const std::vector<Eigen::VectorXd> &IpoptInterface::get_xs() const { return xs_; }

const std::vector<Eigen::VectorXd> &IpoptInterface::get_us() const { return us_; }

const boost::shared_ptr<ShootingProblem> &IpoptInterface::get_problem() const { return problem_; }

}  // namespace crocoddyl

#endif