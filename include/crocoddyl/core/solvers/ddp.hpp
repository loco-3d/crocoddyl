///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_SOLVERS_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_DDP_HPP_

#include <crocoddyl/core/solver-base.hpp>
#include <Eigen/Cholesky>
//TODO: SolverDDP

namespace crocoddyl {

class SolverDDP : public SolverAbstract {
 public:
  SolverDDP(ShootingProblem& problem) : SolverAbstract(problem), regFactor(10.),
    regMin(1e-9), regMax(1e9), cost_try(0.), th_grad(1e-12), th_step(0.5), wasFeasible(false) {
    allocateData();

    const unsigned int& n_alphas = 10;
    alphas.resize(n_alphas);
    for (unsigned int n = 0; n < n_alphas; ++n) {
      alphas[n] = 1. / pow(2., (double) n);
    }
  }
  ~SolverDDP() { }

  bool solve(const std::vector<Eigen::VectorXd>& init_xs,
             const std::vector<Eigen::VectorXd>& init_us,
             const unsigned int& maxiter=100,
             const bool& _isFeasible=false,
             const double& regInit=NAN) override {
    setCandidate(init_xs, init_us, _isFeasible);

    if (std::isnan(regInit)) {
      x_reg = regMin;
      u_reg = regMin;
    } else {
      x_reg = regInit;
      u_reg = regInit;
    }
    wasFeasible = false;

    for (unsigned int iter = 0; iter < maxiter; ++iter) {
      bool recalc = true;
      while (true) {
        try {
          computeDirection(recalc);
        } catch (const char* msg) {
          recalc = false;
          if (x_reg == regMax) {
            return false;
          } else {
            continue;
          }
        }
        break;
      }
      expectedImprovement();

      for (std::vector<double>::const_iterator it = alphas.begin(); it != alphas.end(); ++it) {
        stepLength = *it;

        try {
          dV = tryStep(stepLength);
        } catch (const char* msg) {
          continue;
        }
        dV_exp = stepLength * (d[0] + 0.5 * stepLength * d[1]);

        if (d[0] < th_grad || !isFeasible || dV > th_acceptStep * dV_exp) {
          wasFeasible = isFeasible;
          setCandidate(xs_try, us_try, true);
          cost = cost_try;
          break;
        }
      }

      if (stepLength > th_step) {
       decreaseRegularization();
      }
      if (stepLength == alphas.back()) {
        increaseRegularization();
        if (x_reg == regMax) {
          return false;
        }
      }
      stoppingCriteria();

      const long unsigned int& n_callbacks = callbacks.size();
      if (n_callbacks != 0) {
        for (long unsigned int c = 0; c < n_callbacks; ++c) {
          CallbackAbstract& callback = *callbacks[c];
          callback(this);
        }
      }

      if (wasFeasible && stop < th_stop) {
        return true;
      }
    }
    return false;
  }

  void computeDirection(const bool& recalc=true) override {
    if (recalc) {
      calc();
    }
    backwardPass();
  }

  double tryStep(const double& stepLength) override {
    forwardPass(stepLength);
    return cost - cost_try;
  }

  double stoppingCriteria() override {
    stop = 0.;
    const long unsigned int& T = this->problem.get_T();
    for (long unsigned int t = 0; t < T; ++t) {
      stop += Qu[t].squaredNorm();
    }
    return stop;
  }

  const Eigen::Vector2d& expectedImprovement() override {
    d = Eigen::Vector2d::Zero();
    const long unsigned int& T = this->problem.get_T();
    for (long unsigned int t = 0; t < T; ++t) {
      d[0] += Qu[t].dot(k[t]);
      d[1] -= k[t].dot(Quu[t] * k[t]);
    }
    return d;
  }

private:
  double calc() {
    cost = problem.calcDiff(xs, us);
    if (!isFeasible) {
      const Eigen::VectorXd& x0 = problem.get_x0();
      problem.runningModels[0]->get_state()->diff(xs[0], x0, gaps[0]);

      const long unsigned int& T = problem.get_T();
      for (unsigned long int t = 0; t < T; ++t) {
        ActionModelAbstract* model = problem.runningModels[t];
        std::shared_ptr<ActionDataAbstract>& d = problem.runningDatas[t];
        model->get_state()->diff(xs[t+1], d->get_xnext(), gaps[t+1]);
      }
    }
    return cost;
  }

  void backwardPass() {
    std::shared_ptr<ActionDataAbstract>& d_T = problem.terminalData;
    Vxx.back() = d_T->get_Lxx();
    Vx.back() = d_T->get_Lx();

    const int& ndx = problem.terminalModel->get_ndx();
    const Eigen::VectorXd& xReg = Eigen::VectorXd::Constant(ndx, x_reg);
    if (!std::isnan(x_reg)) {
      Vxx.back().diagonal() += xReg;
    }

    for (long unsigned int t = problem.get_T() - 1; t >= 0; --t) {
      ActionModelAbstract* m = problem.runningModels[t];
      std::shared_ptr<ActionDataAbstract>& d = problem.runningDatas[t];
      const Eigen::MatrixXd& Vxx_p = Vxx[t + 1];
      const Eigen::VectorXd& Vx_p = Vx[t + 1];
      const Eigen::VectorXd& gap_p = gaps[t + 1];

      const Eigen::MatrixXd& FxTVxx_p = d->get_Fx().transpose() * Vxx_p;
      Qxx[t] = d->get_Lxx() + FxTVxx_p * d->get_Fx();
      Qxu[t] = d->get_Lxu() + FxTVxx_p * d->get_Fu();
      Quu[t].noalias() = d->get_Luu() + d->get_Fu().transpose() * Vxx_p * d->get_Fu();
      if (!isFeasible) {
        // In case the xt+1 are not f(xt,ut) i.e warm start not obtained from roll-out.
        const Eigen::VectorXd& relinearization = Vxx_p * gap_p;
        Qx[t] = d->get_Lx() + d->get_Fx().transpose() * Vx_p + d->get_Fx().transpose() * relinearization;
        Qu[t] = d->get_Lu() + d->get_Fu().transpose() * Vx_p + d->get_Fu().transpose() * relinearization;
      } else {
        Qx[t] = d->get_Lx() + d->get_Fx().transpose() * Vx_p;
        Qu[t] = d->get_Lu() + d->get_Fu().transpose() * Vx_p;
      }

      if (!std::isnan(u_reg)) {
        const int& nu = m->get_nu();
        Quu[t].diagonal() += Eigen::VectorXd::Constant(nu, u_reg);
      }

      computeGains(t);

      if (std::isnan(u_reg)) {
        Vx[t] = Qx[t] - K[t].transpose() * Qu[t];
      } else {
        Vx[t] = Qx[t] + K[t].transpose() * (Quu[t] * k[t] - 2. * Qu[t]);
      }
      Vxx[t] = Qxx[t] - Qxu[t] * K[t];
      Vxx[t] = 0.5 * (Vxx[t] + Vxx[t].transpose());// TODO: as suggested by Nicolas

      if (!std::isnan(x_reg)) {
        Vxx[t].diagonal() += xReg;
      }

      const double& Vx_value = Vx[t].sum();
      const double& Vxx_value = Vxx[t].sum();
      if (std::isnan(Vx_value) || std::isnan(Vxx_value)) {
        throw "backward error";
      }
    }
  }

  void forwardPass(const double& stepLength) {
    cost_try = 0.;
    const long unsigned int& T = problem.get_T();
    for (long unsigned int t = 0; t < T; ++t) {
      ActionModelAbstract* m = problem.runningModels[t];
      std::shared_ptr<ActionDataAbstract>& d = problem.runningDatas[t];

      m->get_state()->diff(xs[t], xs_try[t], dx[t]);
      us_try[t] = us[t] - k[t] * stepLength - K[t] * dx[t];
      m->calc(d, xs_try[t], us_try[t]);
      xs_try[t+1] = d->get_xnext();
      cost_try += d->cost;

      const double& value = xs_try[t+1].sum();
      if (std::isnan(value) || std::isinf(value) ||
          std::isnan(cost_try) || std::isnan(cost_try)) {
        throw "forward error";
      }
    }

    ActionModelAbstract* m = problem.terminalModel;
    std::shared_ptr<ActionDataAbstract>& d = problem.terminalData;
    m->calc(d, xs_try.back());
    cost_try += d->cost;

    if (std::isnan(cost_try) || std::isnan(cost_try)) {
      throw "forward error";
    }
  }

  void computeGains(const long unsigned int& t) {
    const Eigen::LLT<Eigen::MatrixXd>& Lb = Quu[t].llt();
    K[t] = Lb.solve(Qxu[t].transpose());
    k[t] = Lb.solve(Qu[t]);
  }

  void increaseRegularization() {
    x_reg *= regFactor;
    if (x_reg > regMax) {
      x_reg = regMax;
    }
    u_reg = x_reg;
  }

  void decreaseRegularization() {
    x_reg /= regFactor;
    if (x_reg < regMin) {
      x_reg = regMin;
    }
    u_reg = x_reg;
  }

  void allocateData() {
    const long unsigned int& T = problem.get_T();
    Vxx.resize(T + 1);
    Vx.resize(T + 1);
    Qxx.resize(T);
    Qxu.resize(T);
    Quu.resize(T);
    Qx.resize(T);
    Qu.resize(T);
    K.resize(T);
    k.resize(T);
    gaps.resize(T + 1);

    xs.resize(T + 1);
    us.resize(T);
    xs_try.resize(T + 1);
    us_try.resize(T);
    dx.resize(T);

    for (long unsigned int t = 0; t < T; ++t) {
      ActionModelAbstract* model = problem.runningModels[t];
      const int& nx = model->get_nx();
      const int& ndx = model->get_ndx();
      const int& nu = model->get_nu();

      Vxx[t] = Eigen::MatrixXd::Zero(ndx, ndx);
      Vx[t] = Eigen::VectorXd::Zero(ndx);
      Qxx[t] = Eigen::MatrixXd::Zero(ndx, ndx);
      Qxu[t] = Eigen::MatrixXd::Zero(ndx, nu);
      Quu[t] = Eigen::MatrixXd::Zero(nu, nu);
      Qx[t] = Eigen::VectorXd::Zero(ndx);
      Qu[t] = Eigen::VectorXd::Zero(nu);
      K[t] = Eigen::MatrixXd::Zero(nu, ndx);
      k[t] = Eigen::VectorXd::Zero(nu);
      gaps[t] = Eigen::VectorXd::Zero(ndx);

      xs[t] = model->get_state()->zero();
      us[t] = Eigen::VectorXd::Zero(nu);
      if (t == 0) {
        xs_try[t] = problem.get_x0();
      } else {
        xs_try[t] = Eigen::VectorXd::Constant(nx, NAN);
      }
      us_try[t] = Eigen::VectorXd::Constant(nu, NAN);
      dx[t] = Eigen::VectorXd::Zero(ndx);
    }
    const int& ndx = problem.terminalModel->get_ndx();
    Vxx.back() = Eigen::MatrixXd::Zero(ndx, ndx);
    Vx.back() = Eigen::VectorXd::Zero(ndx);
    xs.back() = problem.terminalModel->get_state()->zero();
    xs_try.back() = problem.terminalModel->get_state()->zero();
    gaps.back() = Eigen::VectorXd::Zero(ndx);
  };

 protected:
  double regFactor;
  double regMin;
  double regMax;
  double cost_try;

  std::vector<Eigen::VectorXd> xs_try;
  std::vector<Eigen::VectorXd> us_try;
  std::vector<Eigen::VectorXd> dx;

  //allocate data
  std::vector<Eigen::MatrixXd> Vxx;
  std::vector<Eigen::VectorXd> Vx;
  std::vector<Eigen::MatrixXd> Qxx;
  std::vector<Eigen::MatrixXd> Qxu;
  std::vector<Eigen::MatrixXd> Quu;
  std::vector<Eigen::VectorXd> Qx;
  std::vector<Eigen::VectorXd> Qu;
  std::vector<Eigen::MatrixXd> K;
  std::vector<Eigen::VectorXd> k;
  std::vector<Eigen::VectorXd> gaps;

 private:
  Eigen::VectorXd x_next;
  std::vector<double> alphas;
  double th_grad;
  double th_step;
  bool wasFeasible;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
