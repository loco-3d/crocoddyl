#include "crocoddyl/core/numdiff/state.hpp"

namespace crocoddyl {

StateNumDiff::StateNumDiff(StateAbstract& state) : StateAbstract(state.get_nx(), state.get_ndx()), state_(state) {
  disturbance_ = 1e-6;
  // disturbance vector
  dx_.resize(ndx_);
  dx_.setZero();
  // State around which to compute the finite integrale jacobians
  x0_.resize(nx_);
  x0_.setZero();
  // State difference around which to compute the finite difference jacobians
  dx0_.resize(ndx_);
  dx0_.setZero();
  // temporary variable needed
  tmp_x_.resize(nx_);
  tmp_x_.setZero();
}

StateNumDiff::~StateNumDiff() {}

Eigen::VectorXd StateNumDiff::zero() { return state_.zero(); }

Eigen::VectorXd StateNumDiff::rand() { return state_.rand(); }

void StateNumDiff::diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                        Eigen::Ref<Eigen::VectorXd> dxout) {
  state_.diff(x0, x1, dxout);
}

void StateNumDiff::integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                             Eigen::Ref<Eigen::VectorXd> xout) {
  state_.integrate(x, dx, xout);
}

void StateNumDiff::Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                         Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                         Jcomponent firstsecond) {
  assert(is_a_Jcomponent(firstsecond) && ("StateNumDiff::Jdiff: firstsecond "
                                          "must be one of the Jcomponent {both, first, second }"));
  dx_.setZero();
  diff(x0, x1, dx0_);
  if (firstsecond == first || firstsecond == both) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x = int(x0, dx)
      integrate(x0, dx_, tmp_x_);
      // Jfirst[:,k] = diff(tmp_x, x1) = diff(int(x0 + dx), x1)
      diff(tmp_x_, x1, Jfirst.col(i));
      // Jfirst[:,k] = Jfirst[:,k] - tmp_dx_, or
      // Jfirst[:,k] = Jfirst[:,k] - diff(x0, x1)
      Jfirst.col(i) -= dx0_;
      dx_(i) = 0.0;
    }
    Jfirst /= disturbance_;
  }
  if (firstsecond == second || firstsecond == both) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");

    Jsecond.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x = int(x1 + dx)
      integrate(x1, dx_, tmp_x_);
      // Jsecond[:,k] = diff(x0, tmp_x) = diff(x0, int(x1 + dx))
      diff(x0, tmp_x_, Jsecond.col(i));
      // Jsecond[:,k] = J[:,k] - tmp_dx_
      // Jsecond[:,k] = Jsecond[:,k] - diff(x0, x1)
      Jsecond.col(i) -= dx0_;
      dx_(i) = 0.0;
    }
    Jsecond /= disturbance_;
  }
}

void StateNumDiff::Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                              Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                              Jcomponent firstsecond) {
  assert((firstsecond == first || firstsecond == second || firstsecond == both) &&
         ("StateNumDiff::Jdiff: firstsecond must be one of the Jcomponent "
          "{both, first, second }"));
  dx_.setZero();
  // x0_ = integrate(x, dx)
  integrate(x, dx, x0_);

  if (firstsecond == first || firstsecond == both) {
    assert(Jfirst.rows() == ndx_ && Jfirst.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");
    Jfirst.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x_ = integrate(x, dx_) = integrate(x, disturbance_vector)
      integrate(x, dx_, tmp_x_);
      // tmp_x_ = integrate(tmp_x_, dx) = integrate(integrate(x, dx_), dx)
      integrate(tmp_x_, dx, tmp_x_);
      // Jfirst[:,i] = diff(x0_, tmp_x_)
      // Jfirst[:,i] = diff( integrate(x, dx), integrate(integrate(x, dx_), dx))
      diff(x0_, tmp_x_, Jfirst.col(i));
      dx_(i) = 0.0;
    }
    Jfirst /= disturbance_;
  }
  if (firstsecond == second || firstsecond == both) {
    assert(Jsecond.rows() == ndx_ && Jsecond.cols() == ndx_ && "StateNumDiff::Jdiff: Jfirst must be of the good size");
    Jsecond.setZero();
    for (unsigned i = 0; i < ndx_; ++i) {
      dx_(i) = disturbance_;
      // tmp_x_ = integrate(x, dx + dx_) = integrate(x, dx + disturbance_vector)
      integrate(x, dx + dx_, tmp_x_);
      // Jsecond[:,i] = diff(x0_, tmp_x_)
      // Jsecond[:,i] = diff( integrate(x, dx), integrate(x, dx_ + dx) )
      diff(x0_, tmp_x_, Jsecond.col(i));
      dx_(i) = 0.0;
    }
    Jsecond /= disturbance_;
  }
}

}  // namespace crocoddyl
