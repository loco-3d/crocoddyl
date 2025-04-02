///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const MatrixXs& A,
                                             const MatrixXs& B,
                                             const MatrixXs& Q,
                                             const MatrixXs& R,
                                             const MatrixXs& N)
    : Base(std::make_shared<StateVector>(A.cols()), B.cols(), 0),
      drift_free_(true),
      updated_lqr_(false) {
  const std::size_t nx = state_->get_nx();
  MatrixXs G = MatrixXs::Zero(ng_, nx + nu_);
  MatrixXs H = MatrixXs::Zero(nh_, nx + nu_);
  VectorXs f = VectorXs::Zero(nx);
  VectorXs q = VectorXs::Zero(nx);
  VectorXs r = VectorXs::Zero(nu_);
  VectorXs g = VectorXs::Zero(ng_);
  VectorXs h = VectorXs::Zero(nh_);
  set_LQR(A, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(
    const MatrixXs& A, const MatrixXs& B, const MatrixXs& Q, const MatrixXs& R,
    const MatrixXs& N, const VectorXs& f, const VectorXs& q, const VectorXs& r)
    : Base(std::make_shared<StateVector>(A.cols()), B.cols(), 0),
      drift_free_(false),
      updated_lqr_(false) {
  const std::size_t nx = state_->get_nx();
  MatrixXs G = MatrixXs::Zero(ng_, nx + nu_);
  MatrixXs H = MatrixXs::Zero(ng_, nx + nu_);
  VectorXs g = VectorXs::Zero(ng_);
  VectorXs h = VectorXs::Zero(nh_);
  set_LQR(A, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(
    const MatrixXs& A, const MatrixXs& B, const MatrixXs& Q, const MatrixXs& R,
    const MatrixXs& N, const MatrixXs& G, const MatrixXs& H, const VectorXs& f,
    const VectorXs& q, const VectorXs& r, const VectorXs& g, const VectorXs& h)
    : Base(std::make_shared<StateVector>(A.cols()), B.cols(), 0, G.rows(),
           H.rows(), G.rows(), H.rows()),
      drift_free_(false),
      updated_lqr_(false) {
  set_LQR(A, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const std::size_t nx,
                                             const std::size_t nu,
                                             const bool drift_free)
    : Base(std::make_shared<StateVector>(nx), nu, 0),
      A_(MatrixXs::Identity(nx, nx)),
      B_(MatrixXs::Identity(nx, nu)),
      Q_(MatrixXs::Identity(nx, nx)),
      R_(MatrixXs::Identity(nu, nu)),
      N_(MatrixXs::Zero(nx, nu)),
      G_(MatrixXs::Zero(0, nx + nu)),
      H_(MatrixXs::Zero(0, nx + nu)),
      f_(drift_free ? VectorXs::Zero(nx) : VectorXs::Ones(nx)),
      q_(VectorXs::Ones(nx)),
      r_(VectorXs::Ones(nu)),
      g_(VectorXs::Zero(0)),
      h_(VectorXs::Zero(0)),
      drift_free_(drift_free) {}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const ActionModelLQRTpl& copy)
    : Base(std::make_shared<StateVector>(copy.get_A().cols()),
           copy.get_B().cols(), 0, copy.get_G().rows(), copy.get_H().rows(),
           copy.get_G().rows(), copy.get_H().rows()),
      drift_free_(false),
      updated_lqr_(false) {
  set_LQR(copy.get_A(), copy.get_B(), copy.get_Q(), copy.get_R(), copy.get_N(),
          copy.get_G(), copy.get_H(), copy.get_f(), copy.get_q(), copy.get_r(),
          copy.get_g(), copy.get_h());
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  data->xnext.noalias() = A_ * x;
  data->xnext.noalias() += B_ * u;
  data->xnext += f_;

  // cost = 0.5 * x^T * Q * x + 0.5 * u^T * R * u + x^T * N * u + q^T * x + r^T
  // * u
  d->Q_x_tmp.noalias() = Q_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Q_x_tmp);
  d->R_u_tmp.noalias() = R_ * u;
  data->cost += Scalar(0.5) * u.dot(d->R_u_tmp);
  d->Q_x_tmp.noalias() = N_ * u;
  data->cost += x.dot(d->Q_x_tmp);
  data->cost += q_.dot(x);
  data->cost += r_.dot(u);

  // constraints
  const std::size_t nx = state_->get_nx();
  data->g.noalias() = G_.leftCols(nx) * x;
  data->g.noalias() += G_.rightCols(nu_) * u;
  data->g += g_;
  data->h.noalias() = H_.leftCols(nx) * x;
  data->h.noalias() += H_.rightCols(nu_) * u;
  data->h += h_;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  d->xnext = x;

  // cost = 0.5 * x^T * Q * x + q^T * x
  d->Q_x_tmp.noalias() = Q_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Q_x_tmp);
  data->cost += q_.dot(x);

  // constraints
  const std::size_t nx = state_->get_nx();
  data->g.noalias() = G_.leftCols(nx) * x;
  data->g += g_;
  data->h.noalias() = H_.leftCols(nx) * x;
  data->h += h_;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }

  const std::size_t nx = state_->get_nx();
  if (!updated_lqr_) {
    data->Fx = A_;
    data->Fu = B_;
    data->Lxx = Q_;
    data->Luu = R_;
    data->Lxu = N_;
    data->Gx = G_.leftCols(nx);
    data->Gu = G_.rightCols(nu_);
    data->Hx = H_.leftCols(nx);
    data->Hu = H_.rightCols(nu_);
    updated_lqr_ = true;
  }
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
  data->Lx.noalias() += N_ * u;
  data->Lu = r_;
  data->Lu.noalias() += N_.transpose() * x;
  data->Lu.noalias() += R_ * u;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t nx = state_->get_nx();
  if (!updated_lqr_) {
    data->Lxx = Q_;
    data->Gx = G_.leftCols(nx);
    data->Hx = H_.leftCols(nx);
    updated_lqr_ = true;
  }
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelLQRTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
ActionModelLQRTpl<NewScalar> ActionModelLQRTpl<Scalar>::cast() const {
  typedef ActionModelLQRTpl<NewScalar> ReturnType;
  ReturnType ret(A_.template cast<NewScalar>(), B_.template cast<NewScalar>(),
                 Q_.template cast<NewScalar>(), R_.template cast<NewScalar>(),
                 N_.template cast<NewScalar>(), G_.template cast<NewScalar>(),
                 H_.template cast<NewScalar>(), f_.template cast<NewScalar>(),
                 q_.template cast<NewScalar>(), r_.template cast<NewScalar>(),
                 g_.template cast<NewScalar>(), h_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
bool ActionModelLQRTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
ActionModelLQRTpl<Scalar> ActionModelLQRTpl<Scalar>::Random(
    const std::size_t nx, const std::size_t nu, const std::size_t ng,
    const std::size_t nh) {
  MatrixXs A = MatrixXs::Random(nx, nx);
  MatrixXs B = MatrixXs::Random(nx, nu);
  MatrixXs L_tmp = MatrixXs::Random(nx + nu, nx + nu);
  MatrixXs L = L_tmp.transpose() * L_tmp;
  const Eigen::Block<MatrixXs> Q = L.topLeftCorner(nx, nx);
  const Eigen::Block<MatrixXs> R = L.bottomRightCorner(nu, nu);
  const Eigen::Block<MatrixXs> N = L.topRightCorner(nx, nu);
  MatrixXs G = MatrixXs::Random(ng, nx + nu);
  MatrixXs H = MatrixXs::Random(nh, nx + nu);
  VectorXs f = VectorXs::Random(nx);
  VectorXs q = VectorXs::Random(nx);
  VectorXs r = VectorXs::Random(nu);
  VectorXs g = VectorXs::Random(ng);
  VectorXs h = VectorXs::Random(nh);
  return ActionModelLQRTpl<Scalar>(A, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelLQR {nx=" << state_->get_nx() << ", nu=" << nu_
     << ", ng=" << ng_ << ", nh=" << nh_ << ", drift_free=" << drift_free_
     << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_A()
    const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_B()
    const {
  return B_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_f()
    const {
  return f_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Q()
    const {
  return Q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_R()
    const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_N()
    const {
  return N_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_G()
    const {
  return G_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_H()
    const {
  return H_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_q()
    const {
  return q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_r()
    const {
  return r_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_g()
    const {
  return g_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_h()
    const {
  return h_;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_LQR(const MatrixXs& A, const MatrixXs& B,
                                        const MatrixXs& Q, const MatrixXs& R,
                                        const MatrixXs& N, const MatrixXs& G,
                                        const MatrixXs& H, const VectorXs& f,
                                        const VectorXs& q, const VectorXs& r,
                                        const VectorXs& g, const VectorXs& h) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(A.rows()) != nx) {
    throw_pretty(
        "Invalid argument: " << "A should be a squared matrix with size " +
                                    std::to_string(nx));
  }
  if (static_cast<std::size_t>(B.rows()) != nx) {
    throw_pretty(
        "Invalid argument: " << "B has wrong dimension (it should have " +
                                    std::to_string(nx) + " rows)");
  }
  if (static_cast<std::size_t>(Q.rows()) != nx ||
      static_cast<std::size_t>(Q.cols()) != nx) {
    throw_pretty("Invalid argument: "
                 << "Q has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(R.rows()) != nu_ ||
      static_cast<std::size_t>(R.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "R has wrong dimension (it should be " +
                                    std::to_string(nu_) + " x " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(N.rows()) != nx ||
      static_cast<std::size_t>(N.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "N has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(G.rows()) != ng_ ||
      static_cast<std::size_t>(G.cols()) != nx + nu_) {
    throw_pretty(
        "Invalid argument: " << "G has wrong dimension (it should be " +
                                    std::to_string(ng_) + " x " +
                                    std::to_string(nx + nu_) + ")");
  }
  if (static_cast<std::size_t>(H.rows()) != nh_ ||
      static_cast<std::size_t>(H.cols()) != nx + nu_) {
    throw_pretty(
        "Invalid argument: " << "H has wrong dimension (it should be " +
                                    std::to_string(nh_) + " x " +
                                    std::to_string(nx + nu_) + ")");
  }
  if (static_cast<std::size_t>(f.size()) != nx) {
    throw_pretty(
        "Invalid argument: " << "f has wrong dimension (it should be " +
                                    std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(q.size()) != nx) {
    throw_pretty(
        "Invalid argument: " << "q has wrong dimension (it should be " +
                                    std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(r.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "r has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(g.size()) != ng_) {
    throw_pretty(
        "Invalid argument: " << "g has wrong dimension (it should be " +
                                    std::to_string(ng_) + ")");
  }
  if (static_cast<std::size_t>(h.size()) != nh_) {
    throw_pretty(
        "Invalid argument: " << "h has wrong dimension (it should be " +
                                    std::to_string(nh_) + ")");
  }
  L_ = MatrixXs::Zero(nx + nu_, nx + nu_);
  L_ << Q, N, N.transpose(), R;
  if (!checkPSD(L_)) {
    throw_pretty("Invalid argument "
                 << "[Q, N; N.T, R] is not positive semi-definite");
  }
  A_ = A;
  B_ = B;
  f_ = f;
  Q_ = Q;
  R_ = R;
  N_ = N;
  G_ = G;
  H_ = H;
  q_ = q;
  r_ = r;
  g_ = g;
  h_ = h;
  updated_lqr_ = false;
}

}  // namespace crocoddyl
