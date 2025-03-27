///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2023, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/ipopt/ipopt-iface.hpp"

#include <cmath>

namespace crocoddyl {

IpoptInterface::IpoptInterface(const std::shared_ptr<ShootingProblem>& problem)
    : problem_(problem) {
  const std::size_t T = problem_->get_T();
  xs_.resize(T + 1);
  us_.resize(T);
  datas_.resize(T + 1);
  ixu_.resize(T + 1);

  nconst_ = 0;
  nvar_ = 0;
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nxi = models[t]->get_state()->get_nx();
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    xs_[t] = models[t]->get_state()->zero();
    us_[t] = Eigen::VectorXd::Zero(nui);
    datas_[t] = createData(nxi, ndxi, nui);
    ixu_[t] = nvar_;
    nconst_ += ndxi;      // T*ndx eq. constraints for dynamics
    nvar_ += ndxi + nui;  // Multiple shooting, states and controls
  }
  ixu_[T] = nvar_;

  // Initial condition
  nconst_ += models[0]->get_state()->get_ndx();

  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_terminalModel();
  const std::size_t nxi = model->get_state()->get_nx();
  const std::size_t ndxi = model->get_state()->get_ndx();
  nvar_ += ndxi;  // final node
  xs_[T] = model->get_state()->zero();
  datas_[T] = createData(nxi, ndxi, 0);
}

void IpoptInterface::resizeData() {
  const std::size_t T = problem_->get_T();
  nvar_ = 0;
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nxi = models[t]->get_state()->get_nx();
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    xs_[t].conservativeResize(nxi);
    us_[t].conservativeResize(nui);
    datas_[t]->resize(nxi, ndxi, nui);
    ixu_[t] = nvar_;
    nconst_ += ndxi;      // T*ndx eq. constraints for dynamics
    nvar_ += ndxi + nui;  // Multiple shooting, states and controls
  }
  ixu_[T] = nvar_;

  // Initial condition
  nconst_ += models[0]->get_state()->get_ndx();

  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_terminalModel();
  const std::size_t nxi = model->get_state()->get_nx();
  const std::size_t ndxi = model->get_state()->get_ndx();
  nvar_ += ndxi;  // final node
  xs_[T].conservativeResize(nxi);
  datas_[T]->resize(nxi, ndxi, 0);
}

IpoptInterface::~IpoptInterface() {}

bool IpoptInterface::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                                  Ipopt::Index& nnz_jac_g,
                                  Ipopt::Index& nnz_h_lag,
                                  IndexStyleEnum& index_style) {
  n = static_cast<Ipopt::Index>(nvar_);    // number of variables
  m = static_cast<Ipopt::Index>(nconst_);  // number of constraints

  nnz_jac_g = 0;  // Jacobian nonzeros for dynamic constraints
  nnz_h_lag = 0;  // Hessian nonzeros (only lower triangular part)
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::size_t T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t ndxi_next =
        t + 1 == T ? problem_->get_terminalModel()->get_state()->get_ndx()
                   : models[t + 1]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    nnz_jac_g += ndxi * (ndxi + ndxi_next + nui);

    // Hessian
    std::size_t nonzero = 0;
    for (std::size_t i = 1; i <= (ndxi + nui); ++i) {
      nonzero += i;
    }
    nnz_h_lag += nonzero;
  }

  // Initial condition
  nnz_jac_g +=
      models[0]->get_state()->get_ndx() * models[0]->get_state()->get_ndx();

  // Hessian nonzero for the terminal cost
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  std::size_t nonzero = 0;
  for (std::size_t i = 1; i <= ndxi; ++i) {
    nonzero += i;
  }
  nnz_h_lag += nonzero;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l,
                                     Ipopt::Number* x_u, Ipopt::Index m,
                                     Ipopt::Number* g_l, Ipopt::Number* g_u) {
#else
bool IpoptInterface::get_bounds_info(Ipopt::Index, Ipopt::Number* x_l,
                                     Ipopt::Number* x_u, Ipopt::Index,
                                     Ipopt::Number* g_l, Ipopt::Number* g_u) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");
  assert_pretty(m == static_cast<Ipopt::Index>(nconst_),
                "Inconsistent number of constraints");

  // Adding bounds
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < problem_->get_T(); ++t) {
    // Running state bounds
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    for (std::size_t j = 0; j < ndxi; ++j) {
      x_l[ixu_[t] + j] = std::numeric_limits<double>::lowest();
      x_u[ixu_[t] + j] = std::numeric_limits<double>::max();
    }
    for (std::size_t j = 0; j < nui; ++j) {
      x_l[ixu_[t] + ndxi + j] = models[t]->get_has_control_limits()
                                    ? models[t]->get_u_lb()(j)
                                    : std::numeric_limits<double>::lowest();
      x_u[ixu_[t] + ndxi + j] = models[t]->get_has_control_limits()
                                    ? models[t]->get_u_ub()(j)
                                    : std::numeric_limits<double>::max();
    }
  }

  // Final state bounds
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  for (std::size_t j = 0; j < ndxi; j++) {
    x_l[ixu_.back() + j] = std::numeric_limits<double>::lowest();
    x_u[ixu_.back() + j] = std::numeric_limits<double>::max();
  }

  // Dynamics & Initial conditions (all equal to zero)
  for (Ipopt::Index i = 0; i < static_cast<Ipopt::Index>(nconst_); ++i) {
    g_l[i] = 0;
    g_u[i] = 0;
  }

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::get_starting_point(Ipopt::Index n, bool init_x,
                                        Ipopt::Number* x, bool init_z,
                                        Ipopt::Number* /*z_L*/,
                                        Ipopt::Number* /*z_U*/, Ipopt::Index m,
                                        bool init_lambda,
                                        Ipopt::Number* /*lambda*/) {
#else
bool IpoptInterface::get_starting_point(Ipopt::Index, bool /*init_x*/,
                                        Ipopt::Number* x, bool, Ipopt::Number*,
                                        Ipopt::Number*, Ipopt::Index, bool,
                                        Ipopt::Number*) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");
  assert_pretty(m == static_cast<Ipopt::Index>(nconst_),
                "Inconsistent number of constraints");
  assert_pretty(init_x == true,
                "Make sure to provide initial value for primal variables");
  assert_pretty(init_z == false,
                "Cannot provide initial value for bound multipliers");
  assert_pretty(init_lambda == false,
                "Cannot provide initial value for constraint multipliers");

  // initialize to the given starting point
  // State variables are always at 0 since they represent increments from the
  // given initial point
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < problem_->get_T(); ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    for (std::size_t j = 0; j < ndxi; ++j) {
      x[ixu_[t] + j] = 0;
    }
    for (std::size_t j = 0; j < nui; ++j) {
      x[ixu_[t] + ndxi + j] = us_[t](j);
    }
  }
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  for (std::size_t j = 0; j < ndxi; j++) {
    x[ixu_.back() + j] = 0;
  }

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::eval_f(Ipopt::Index n, const Ipopt::Number* x,
                            bool /*new_x*/, Ipopt::Number& obj_value) {
#else
bool IpoptInterface::eval_f(Ipopt::Index, const Ipopt::Number* x, bool,
                            Ipopt::Number& obj_value) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");

  // Running costs
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  const std::size_t T = problem_->get_T();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();

    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu_[t], ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu_[t] + ndxi, nui);
    model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    model->calc(data, datas_[t]->x, datas_[t]->u);
  }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp simd reduction(+ : obj_value)
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    obj_value += data->cost;
  }

  // Terminal costs
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& data =
      problem_->get_terminalData();
  const std::size_t ndxi = model->get_state()->get_ndx();

  datas_[T]->dx = Eigen::VectorXd::Map(x + ixu_.back(), ndxi);
  model->get_state()->integrate(xs_[T], datas_[T]->dx, datas_[T]->x);
  model->calc(data, datas_[T]->x);
  obj_value += data->cost;

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::eval_grad_f(Ipopt::Index n, const Ipopt::Number* x,
                                 bool /*new_x*/, Ipopt::Number* grad_f) {
#else
bool IpoptInterface::eval_grad_f(Ipopt::Index, const Ipopt::Number* x, bool,
                                 Ipopt::Number* grad_f) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  const std::size_t T = problem_->get_T();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();

    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu_[t], ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu_[t] + ndxi, nui);
    model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    model->get_state()->Jintegrate(xs_[t], datas_[t]->dx, datas_[t]->Jint_dx,
                                   datas_[t]->Jint_dx, second, setto);
    model->calc(data, datas_[t]->x, datas_[t]->u);
    model->calcDiff(data, datas_[t]->x, datas_[t]->u);
    datas_[t]->Ldx.noalias() = datas_[t]->Jint_dx.transpose() * data->Lx;
  }
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();
    for (std::size_t j = 0; j < ndxi; ++j) {
      grad_f[ixu_[t] + j] = datas_[t]->Ldx(j);
    }
    for (std::size_t j = 0; j < nui; ++j) {
      grad_f[ixu_[t] + ndxi + j] = data->Lu(j);
    }
  }

  // Terminal model
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& data =
      problem_->get_terminalData();
  const std::size_t ndxi = model->get_state()->get_ndx();

  datas_[T]->dx = Eigen::VectorXd::Map(x + ixu_.back(), ndxi);
  model->get_state()->integrate(xs_[T], datas_[T]->dx, datas_[T]->x);
  model->get_state()->Jintegrate(xs_[T], datas_[T]->dx, datas_[T]->Jint_dx,
                                 datas_[T]->Jint_dx, second, setto);
  model->calc(data, datas_[T]->x);
  model->calcDiff(data, datas_[T]->x);
  datas_[T]->Ldx.noalias() = datas_[T]->Jint_dx.transpose() * data->Lx;
  for (std::size_t j = 0; j < ndxi; ++j) {
    grad_f[ixu_.back() + j] = datas_[T]->Ldx(j);
  }

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::eval_g(Ipopt::Index n, const Ipopt::Number* x,
                            bool /*new_x*/, Ipopt::Index m, Ipopt::Number* g) {
#else
bool IpoptInterface::eval_g(Ipopt::Index, const Ipopt::Number* x,
                            bool /*new_x*/, Ipopt::Index, Ipopt::Number* g) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");
  assert_pretty(m == static_cast<Ipopt::Index>(nconst_),
                "Inconsistent number of constraints");

  // Dynamic constraints
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  const std::size_t T = problem_->get_T();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    const std::size_t nui = model->get_nu();
    const std::shared_ptr<ActionModelAbstract>& model_next =
        t + 1 == T ? problem_->get_terminalModel() : models[t + 1];
    const std::size_t ndxi_next = model_next->get_state()->get_ndx();

    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu_[t], ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu_[t] + ndxi, nui);
    datas_[t]->dxnext =
        Eigen::VectorXd::Map(x + ixu_[t] + ndxi + nui, ndxi_next);
    model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    model_next->get_state()->integrate(xs_[t + 1], datas_[t]->dxnext,
                                       datas_[t]->xnext);
    model->calc(data, datas_[t]->x, datas_[t]->u);
    model->get_state()->diff(data->xnext, datas_[t]->xnext, datas_[t]->x_diff);
  }

  std::size_t ix = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t ndxi = model->get_state()->get_ndx();
    for (std::size_t j = 0; j < ndxi; ++j) {
      g[ix + j] = datas_[t]->x_diff[j];
    }
    ix += ndxi;
  }

  // Initial conditions
  const std::shared_ptr<ActionModelAbstract>& model = models[0];
  const std::size_t ndxi = model->get_state()->get_ndx();
  datas_[0]->dx = Eigen::VectorXd::Map(x, ndxi);
  model->get_state()->integrate(xs_[0], datas_[0]->dx, datas_[0]->x);
  model->get_state()->diff(datas_[0]->x, problem_->get_x0(),
                           datas_[0]->x_diff);  // x(0) - x_0
  for (std::size_t j = 0; j < ndxi; j++) {
    g[ix + j] = datas_[0]->x_diff[j];
  }

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::eval_jac_g(Ipopt::Index n, const Ipopt::Number* x,
                                bool /*new_x*/, Ipopt::Index m,
                                Ipopt::Index nele_jac, Ipopt::Index* iRow,
                                Ipopt::Index* jCol, Ipopt::Number* values) {
#else
bool IpoptInterface::eval_jac_g(Ipopt::Index, const Ipopt::Number* x, bool,
                                Ipopt::Index, Ipopt::Index, Ipopt::Index* iRow,
                                Ipopt::Index* jCol, Ipopt::Number* values) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");
  assert_pretty(m == static_cast<Ipopt::Index>(nconst_),
                "Inconsistent number of constraints");

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  if (values == NULL) {
    // Dynamic constraints
    std::size_t idx = 0;
    std::size_t ix = 0;
    const std::size_t T = problem_->get_T();
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();
      const std::size_t ndxi_next =
          t + 1 == T ? problem_->get_terminalModel()->get_state()->get_ndx()
                     : models[t + 1]->get_state()->get_ndx();
      for (std::size_t idx_row = 0; idx_row < ndxi; ++idx_row) {
        for (std::size_t idx_col = 0; idx_col < (ndxi + nui + ndxi_next);
             ++idx_col) {
          iRow[idx] = static_cast<Ipopt::Index>(ix + idx_row);
          jCol[idx] = static_cast<Ipopt::Index>(ixu_[t] + idx_col);
          idx++;
        }
      }
      ix += ndxi;
    }

    // Initial condition
    const std::size_t ndxi = models[0]->get_state()->get_ndx();
    for (std::size_t idx_row = 0; idx_row < ndxi; ++idx_row) {
      for (std::size_t idx_col = 0; idx_col < ndxi; ++idx_col) {
        iRow[idx] = static_cast<Ipopt::Index>(ix + idx_row);
        jCol[idx] = static_cast<Ipopt::Index>(idx_col);
        idx++;
      }
    }

    assert_pretty(nele_jac == static_cast<Ipopt::Index>(idx),
                  "Number of jacobian elements set does not coincide with the "
                  "total non-zero Jacobian values");
  } else {
    const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
        problem_->get_runningDatas();
    // Dynamic constraints
    const std::size_t T = problem_->get_T();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::shared_ptr<ActionDataAbstract>& data = datas[t];
      const std::shared_ptr<ActionModelAbstract>& model_next =
          t + 1 == T ? problem_->get_terminalModel() : models[t + 1];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t ndxi_next = model_next->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();
      datas_[t]->dx = Eigen::VectorXd::Map(x + ixu_[t], ndxi);
      datas_[t]->u = Eigen::VectorXd::Map(x + ixu_[t] + ndxi, nui);
      datas_[t]->dxnext =
          Eigen::VectorXd::Map(x + ixu_[t] + ndxi + nui, ndxi_next);

      model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
      model_next->get_state()->integrate(xs_[t + 1], datas_[t]->dxnext,
                                         datas_[t]->xnext);
      model->calcDiff(data, datas_[t]->x, datas_[t]->u);
      model_next->get_state()->Jintegrate(
          xs_[t + 1], datas_[t]->dxnext, datas_[t]->Jint_dxnext,
          datas_[t]->Jint_dxnext, second,
          setto);  // datas_[t]->Jsum_dxnext == eq. 81
      model->get_state()->Jdiff(
          data->xnext, datas_[t]->xnext, datas_[t]->Jdiff_x,
          datas_[t]->Jdiff_xnext,
          both);  // datas_[t+1]->Jdiff_x == eq. 83, datas_[t]->Jdiff_x == eq.82
      model->get_state()->Jintegrate(xs_[t], datas_[t]->dx, datas_[t]->Jint_dx,
                                     datas_[t]->Jint_dx, second,
                                     setto);  // datas_[t]->Jsum_dx == eq. 81
      datas_[t]->Jg_dxnext.noalias() =
          datas_[t]->Jdiff_xnext * datas_[t]->Jint_dxnext;  // chain rule
      datas_[t]->FxJint_dx.noalias() = data->Fx * datas_[t]->Jint_dx;
      datas_[t]->Jg_dx.noalias() = datas_[t]->Jdiff_x * datas_[t]->FxJint_dx;
      datas_[t]->Jg_u.noalias() = datas_[t]->Jdiff_x * data->Fu;
    }
    std::size_t idx = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::shared_ptr<ActionModelAbstract>& model_next =
          t + 1 == T ? problem_->get_terminalModel() : models[t + 1];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();
      const std::size_t ndxi_next = model_next->get_state()->get_ndx();
      for (std::size_t idx_row = 0; idx_row < ndxi; ++idx_row) {
        for (std::size_t idx_col = 0; idx_col < ndxi; ++idx_col) {
          values[idx] = datas_[t]->Jg_dx(idx_row, idx_col);
          idx++;
        }
        for (std::size_t idx_col = 0; idx_col < nui; ++idx_col) {
          values[idx] = datas_[t]->Jg_u(idx_row, idx_col);
          idx++;
        }
        // This could be more optimized since there are a lot of zeros!
        for (std::size_t idx_col = 0; idx_col < ndxi_next; ++idx_col) {
          values[idx] = datas_[t]->Jg_dxnext(idx_row, idx_col);
          idx++;
        }
      }
    }

    // Initial condition
    const std::shared_ptr<ActionModelAbstract>& model = models[0];
    const std::size_t ndxi = model->get_state()->get_ndx();
    datas_[0]->dx = Eigen::VectorXd::Map(x, ndxi);

    model->get_state()->integrate(xs_[0], datas_[0]->dx, datas_[0]->x);
    model->get_state()->Jdiff(datas_[0]->x, problem_->get_x0(),
                              datas_[0]->Jdiff_x, datas_[0]->Jdiff_x, first);
    model->get_state()->Jintegrate(xs_[0], datas_[0]->dx, datas_[0]->Jint_dx,
                                   datas_[0]->Jint_dx, second, setto);
    datas_[0]->Jg_ic.noalias() = datas_[0]->Jdiff_x * datas_[0]->Jint_dx;
    for (std::size_t idx_row = 0; idx_row < ndxi; ++idx_row) {
      for (std::size_t idx_col = 0; idx_col < ndxi; ++idx_col) {
        values[idx] = datas_[0]->Jg_ic(idx_row, idx_col);
        idx++;
      }
    }
  }

  return true;
}

#ifndef NDEBUG
bool IpoptInterface::eval_h(Ipopt::Index n, const Ipopt::Number* x,
                            bool /*new_x*/, Ipopt::Number obj_factor,
                            Ipopt::Index m, const Ipopt::Number* /*lambda*/,
                            bool /*new_lambda*/, Ipopt::Index nele_hess,
                            Ipopt::Index* iRow, Ipopt::Index* jCol,
                            Ipopt::Number* values) {
#else
bool IpoptInterface::eval_h(Ipopt::Index, const Ipopt::Number* x, bool,
                            Ipopt::Number obj_factor, Ipopt::Index,
                            const Ipopt::Number*, bool, Ipopt::Index,
                            Ipopt::Index* iRow, Ipopt::Index* jCol,
                            Ipopt::Number* values) {
#endif
  assert_pretty(n == static_cast<Ipopt::Index>(nvar_),
                "Inconsistent number of decision variables");
  assert_pretty(m == static_cast<Ipopt::Index>(nconst_),
                "Inconsistent number of constraints");

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::size_t T = problem_->get_T();
  if (values == NULL) {
    // return the structure. This is a symmetric matrix, fill the lower left
    // triangle only

    // Running Costs
    std::size_t idx = 0;
    for (std::size_t t = 0; t < problem_->get_T(); ++t) {
      const std::shared_ptr<ActionModelAbstract> model = models[t];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();
      for (std::size_t idx_row = 0; idx_row < ndxi + nui; ++idx_row) {
        for (std::size_t idx_col = 0; idx_col < ndxi + nui; ++idx_col) {
          // We need the lower triangular matrix
          if (idx_col > idx_row) {
            break;
          }
          iRow[idx] = static_cast<Ipopt::Index>(ixu_[t] + idx_row);
          jCol[idx] = static_cast<Ipopt::Index>(ixu_[t] + idx_col);
          idx++;
        }
      }
    }

    // Terminal costs
    const std::size_t ndxi =
        problem_->get_terminalModel()->get_state()->get_ndx();
    for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
      for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
        // We need the lower triangular matrix
        if (idx_col > idx_row) {
          break;
        }
        iRow[idx] = static_cast<Ipopt::Index>(ixu_.back() + idx_row);
        jCol[idx] = static_cast<Ipopt::Index>(ixu_.back() + idx_col);
        idx++;
      }
    }

    assert_pretty(nele_hess == static_cast<Ipopt::Index>(idx),
                  "Number of Hessian elements set does not coincide with the "
                  "total non-zero Hessian values");
  } else {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only
    // Running Costs
    const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
        problem_->get_runningDatas();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::shared_ptr<ActionDataAbstract>& data = datas[t];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();
      datas_[t]->dx = Eigen::VectorXd::Map(x + ixu_[t], ndxi);
      datas_[t]->u = Eigen::VectorXd::Map(x + ixu_[t] + ndxi, nui);

      model->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
      model->calcDiff(data, datas_[t]->x,
                      datas_[t]->u);  // this might be removed
      model->get_state()->Jintegrate(xs_[t], datas_[t]->dx, datas_[t]->Jint_dx,
                                     datas_[t]->Jint_dx, second, setto);
      datas_[t]->Ldxdx.noalias() =
          datas_[t]->Jint_dx.transpose() * data->Lxx * datas_[t]->Jint_dx;
      datas_[t]->Ldxu.noalias() = datas_[t]->Jint_dx.transpose() * data->Lxu;
    }

    std::size_t idx = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::shared_ptr<ActionDataAbstract>& data = datas[t];
      const std::size_t ndxi = model->get_state()->get_ndx();
      const std::size_t nui = model->get_nu();
      for (std::size_t idx_row = 0; idx_row < ndxi; ++idx_row) {
        for (std::size_t idx_col = 0; idx_col < ndxi; ++idx_col) {
          // We need the lower triangular matrix
          if (idx_col > idx_row) {
            break;
          }
          values[idx] = obj_factor * datas_[t]->Ldxdx(idx_row, idx_col);
          idx++;
        }
      }
      for (std::size_t idx_row = 0; idx_row < nui; ++idx_row) {
        for (std::size_t idx_col = 0; idx_col < ndxi; ++idx_col) {
          values[idx] = obj_factor * datas_[t]->Ldxu(idx_col, idx_row);
          idx++;
        }
        for (std::size_t idx_col = 0; idx_col < nui; ++idx_col) {
          if (idx_col > idx_row) {
            break;
          }
          values[idx] = obj_factor * data->Luu(idx_row, idx_col);
          idx++;
        }
      }
    }

    // Terminal costs
    const std::shared_ptr<ActionModelAbstract>& model =
        problem_->get_terminalModel();
    const std::shared_ptr<ActionDataAbstract>& data =
        problem_->get_terminalData();
    const std::size_t ndxi = model->get_state()->get_ndx();
    datas_[T]->dx = Eigen::VectorXd::Map(x + ixu_.back(), ndxi);

    model->get_state()->integrate(xs_[T], datas_[T]->dx, datas_[T]->x);
    model->calc(data, datas_[T]->x);
    model->calcDiff(data, datas_[T]->x);
    model->get_state()->Jintegrate(xs_[T], datas_[T]->dx, datas_[T]->Jint_dx,
                                   datas_[T]->Jint_dx, second, setto);
    datas_[T]->Ldxdx.noalias() =
        datas_[T]->Jint_dx.transpose() * data->Lxx * datas_[T]->Jint_dx;
    for (std::size_t idx_row = 0; idx_row < ndxi; idx_row++) {
      for (std::size_t idx_col = 0; idx_col < ndxi; idx_col++) {
        // We need the lower triangular matrix
        if (idx_col > idx_row) {
          break;
        }
        values[idx] = datas_[T]->Ldxdx(idx_row, idx_col);
        idx++;
      }
    }
  }

  return true;
}

void IpoptInterface::finalize_solution(
    Ipopt::SolverReturn /*status*/, Ipopt::Index /*n*/, const Ipopt::Number* x,
    const Ipopt::Number* /*z_L*/, const Ipopt::Number* /*z_U*/,
    Ipopt::Index /*m*/, const Ipopt::Number* /*g*/,
    const Ipopt::Number* /*lambda*/, Ipopt::Number obj_value,
    const Ipopt::IpoptData* /*ip_data*/,
    Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) {
  // Copy the solution to vector once solver is finished
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::size_t T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    datas_[t]->dx = Eigen::VectorXd::Map(x + ixu_[t], ndxi);
    datas_[t]->u = Eigen::VectorXd::Map(x + ixu_[t] + ndxi, nui);

    models[t]->get_state()->integrate(xs_[t], datas_[t]->dx, datas_[t]->x);
    xs_[t] = datas_[t]->x;
    us_[t] = datas_[t]->u;
  }
  // Terminal node
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_terminalModel();
  const std::size_t ndxi = model->get_state()->get_ndx();
  datas_[T]->dx = Eigen::VectorXd::Map(x + ixu_.back(), ndxi);
  model->get_state()->integrate(xs_[T], datas_[T]->dx, datas_[T]->x);
  xs_[T] = datas_[T]->x;

  cost_ = obj_value;
}

bool IpoptInterface::intermediate_callback(
    Ipopt::AlgorithmMode /*mode*/, Ipopt::Index /*iter*/,
    Ipopt::Number /*obj_value*/, Ipopt::Number /*inf_pr*/,
    Ipopt::Number /*inf_du*/, Ipopt::Number /*mu*/, Ipopt::Number /*d_norm*/,
    Ipopt::Number /*regularization_size*/, Ipopt::Number /*alpha_du*/,
    Ipopt::Number /*alpha_pr*/, Ipopt::Index /*ls_trials*/,
    const Ipopt::IpoptData* /*ip_data*/,
    Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) {
  return true;
}

std::shared_ptr<IpoptInterfaceData> IpoptInterface::createData(
    const std::size_t nx, const std::size_t ndx, const std::size_t nu) {
  return std::allocate_shared<IpoptInterfaceData>(
      Eigen::aligned_allocator<IpoptInterfaceData>(), nx, ndx, nu);
}

void IpoptInterface::set_xs(const std::vector<Eigen::VectorXd>& xs) {
  xs_ = xs;
}

void IpoptInterface::set_us(const std::vector<Eigen::VectorXd>& us) {
  us_ = us;
}

std::size_t IpoptInterface::get_nvar() const { return nvar_; }

std::size_t IpoptInterface::get_nconst() const { return nconst_; }

const std::vector<Eigen::VectorXd>& IpoptInterface::get_xs() const {
  return xs_;
}

const std::vector<Eigen::VectorXd>& IpoptInterface::get_us() const {
  return us_;
}

const std::shared_ptr<ShootingProblem>& IpoptInterface::get_problem() const {
  return problem_;
}

double IpoptInterface::get_cost() const { return cost_; }

}  // namespace crocoddyl
