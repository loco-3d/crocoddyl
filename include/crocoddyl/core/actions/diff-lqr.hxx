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
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(
    const MatrixXs& Aq, const MatrixXs& Av, const MatrixXs& B,
    const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N)
    : Base(std::make_shared<StateVector>(2 * Aq.cols()), B.cols(), 0),
      drift_free_(true),
      updated_lqr_(false) {
  const std::size_t nq = state_->get_nq();
  MatrixXs G = MatrixXs::Zero(ng_, 2 * nq + nu_);
  MatrixXs H = MatrixXs::Zero(nh_, 2 * nq + nu_);
  VectorXs f = VectorXs::Zero(nq);
  VectorXs q = VectorXs::Zero(2 * nq);
  VectorXs r = VectorXs::Zero(nu_);
  VectorXs g = VectorXs::Zero(ng_);
  VectorXs h = VectorXs::Zero(nh_);
  set_LQR(Aq, Av, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(
    const MatrixXs& Aq, const MatrixXs& Av, const MatrixXs& B,
    const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N, const VectorXs& f,
    const VectorXs& q, const VectorXs& r)
    : Base(std::make_shared<StateVector>(2 * Aq.cols()), B.cols(), 0),
      drift_free_(false),
      updated_lqr_(false) {
  const std::size_t nq = state_->get_nq();
  MatrixXs G = MatrixXs::Zero(ng_, 2 * nq + nu_);
  MatrixXs H = MatrixXs::Zero(ng_, 2 * nq + nu_);
  VectorXs g = VectorXs::Zero(ng_);
  VectorXs h = VectorXs::Zero(nh_);
  set_LQR(Aq, Av, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(
    const MatrixXs& Aq, const MatrixXs& Av, const MatrixXs& B,
    const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N, const MatrixXs& G,
    const MatrixXs& H, const VectorXs& f, const VectorXs& q, const VectorXs& r,
    const VectorXs& g, const VectorXs& h)
    : Base(std::make_shared<StateVector>(2 * Aq.cols()), B.cols(), 0, G.rows(),
           H.rows(), G.rows(), H.rows()),
      drift_free_(false),
      updated_lqr_(false) {
  set_LQR(Aq, Av, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(
    const std::size_t nq, const std::size_t nu, const bool drift_free)
    : Base(std::make_shared<StateVector>(2 * nq), nu),
      Aq_(MatrixXs::Identity(nq, nq)),
      Av_(MatrixXs::Identity(nq, nq)),
      B_(MatrixXs::Identity(nq, nu)),
      Q_(MatrixXs::Identity(2 * nq, 2 * nq)),
      R_(MatrixXs::Identity(nu, nu)),
      N_(MatrixXs::Zero(2 * nq, nu)),
      G_(MatrixXs::Zero(0, 2 * nq + nu)),
      H_(MatrixXs::Zero(0, 2 * nq + nu)),
      f_(drift_free ? VectorXs::Zero(nq) : VectorXs::Ones(nq)),
      q_(VectorXs::Ones(2 * nq)),
      r_(VectorXs::Ones(nu)),
      g_(VectorXs::Zero(0)),
      h_(VectorXs::Zero(0)),
      drift_free_(drift_free),
      updated_lqr_(false) {}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>::DifferentialActionModelLQRTpl(
    const DifferentialActionModelLQRTpl& copy)
    : Base(std::make_shared<StateVector>(2 * copy.get_Aq().cols()),
           copy.get_B().cols(), 0, copy.get_G().rows(), copy.get_H().rows(),
           copy.get_G().rows(), copy.get_H().rows()),
      drift_free_(false),
      updated_lqr_(false) {
  set_LQR(copy.get_Aq(), copy.get_Av(), copy.get_B(), copy.get_Q(),
          copy.get_R(), copy.get_N(), copy.get_G(), copy.get_H(), copy.get_f(),
          copy.get_q(), copy.get_r(), copy.get_g(), copy.get_h());
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
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
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  data->xout.noalias() = Aq_ * q;
  data->xout.noalias() += Av_ * v;
  data->xout.noalias() += B_ * u;
  data->xout += f_;

  // cost = 0.5 * x^T * Q * x + 0.5 * u^T * R * u + x^T * N * u + q^T * x + r^T
  // * u
  data->cost = Scalar(0.5) * x.dot(Q_ * x);
  data->cost += Scalar(0.5) * u.dot(R_ * u);
  data->cost += x.dot(N_ * u);
  data->cost += q_.dot(x);
  data->cost += r_.dot(u);

  // constraints
  const std::size_t nq = state_->get_nq();
  data->g.noalias() = G_.leftCols(nq) * q;
  data->g.noalias() += G_.middleCols(nq, nq) * v;
  data->g.noalias() += G_.rightCols(nu_) * u;
  data->g += g_;
  data->h.noalias() = H_.leftCols(nq) * q;
  data->h.noalias() += H_.middleCols(nq, nq) * v;
  data->h.noalias() += H_.rightCols(nu_) * u;
  data->h += h_;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  // cost = 0.5 * x^T * Q * x + q^T * x
  data->cost = Scalar(0.5) * x.dot(Q_ * x);
  data->cost += q_.dot(x);

  // constraints
  const std::size_t nq = state_->get_nq();
  data->g.noalias() = G_.leftCols(nq) * q;
  data->g.noalias() += G_.middleCols(nq, nq) * v;
  data->g += g_;
  data->h.noalias() = H_.leftCols(nq) * q;
  data->h.noalias() += H_.middleCols(nq, nq) * v;
  data->h += h_;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
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

  const std::size_t nq = state_->get_nq();
  if (!updated_lqr_) {
    data->Fx.leftCols(nq) = Aq_;
    data->Fx.rightCols(nq) = Av_;
    data->Fu = B_;
    data->Lxx = Q_;
    data->Luu = R_;
    data->Lxu = N_;
    data->Gx = G_.leftCols(2 * nq);
    data->Gu = G_.rightCols(nu_);
    data->Hx = H_.leftCols(2 * nq);
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
void DifferentialActionModelLQRTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t nq = state_->get_nq();
  if (!updated_lqr_) {
    data->Lxx = Q_;
    data->Gx = G_.leftCols(2 * nq);
    data->Hx = H_.leftCols(2 * nq);
    updated_lqr_ = true;
  }
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelLQRTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelLQRTpl<NewScalar>
DifferentialActionModelLQRTpl<Scalar>::cast() const {
  typedef DifferentialActionModelLQRTpl<NewScalar> ReturnType;
  ReturnType ret(Aq_.template cast<NewScalar>(), Av_.template cast<NewScalar>(),
                 B_.template cast<NewScalar>(), Q_.template cast<NewScalar>(),
                 R_.template cast<NewScalar>(), N_.template cast<NewScalar>(),
                 G_.template cast<NewScalar>(), H_.template cast<NewScalar>(),
                 f_.template cast<NewScalar>(), q_.template cast<NewScalar>(),
                 r_.template cast<NewScalar>(), g_.template cast<NewScalar>(),
                 h_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
bool DifferentialActionModelLQRTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
DifferentialActionModelLQRTpl<Scalar>
DifferentialActionModelLQRTpl<Scalar>::Random(const std::size_t nq,
                                              const std::size_t nu,
                                              const std::size_t ng,
                                              const std::size_t nh) {
  MatrixXs Aq = MatrixXs::Random(nq, nq);
  MatrixXs Av = MatrixXs::Random(nq, nq);
  MatrixXs B = MatrixXs::Random(nq, nu);
  MatrixXs L_tmp = MatrixXs::Random(2 * nq + nu, 2 * nq + nu);
  MatrixXs L = L_tmp.transpose() * L_tmp;
  const Eigen::Block<MatrixXs> Q = L.topLeftCorner(2 * nq, 2 * nq);
  const Eigen::Block<MatrixXs> R = L.bottomRightCorner(nu, nu);
  const Eigen::Block<MatrixXs> N = L.topRightCorner(2 * nq, nu);
  MatrixXs G = MatrixXs::Random(ng, 2 * nq + nu);
  MatrixXs H = MatrixXs::Random(nh, 2 * nq + nu);
  VectorXs f = VectorXs::Random(nq);
  VectorXs q = VectorXs::Random(2 * nq);
  VectorXs r = VectorXs::Random(nu);
  VectorXs g = VectorXs::Random(ng);
  VectorXs h = VectorXs::Random(nh);
  return DifferentialActionModelLQRTpl<Scalar>(Aq, Av, B, Q, R, N, G, H, f, q,
                                               r, g, h);
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelLQR {nq=" << state_->get_nq() << ", nu=" << nu_
     << ", ng=" << ng_ << ", nh=" << nh_ << ", drift_free=" << drift_free_
     << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Aq() const {
  return Aq_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Av() const {
  return Av_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_B() const {
  return B_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_f() const {
  return f_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_Q() const {
  return Q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_R() const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_N() const {
  return N_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_G() const {
  return G_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
DifferentialActionModelLQRTpl<Scalar>::get_H() const {
  return H_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_q() const {
  return q_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_r() const {
  return r_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_g() const {
  return g_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelLQRTpl<Scalar>::get_h() const {
  return h_;
}

template <typename Scalar>
void DifferentialActionModelLQRTpl<Scalar>::set_LQR(
    const MatrixXs& Aq, const MatrixXs& Av, const MatrixXs& B,
    const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N, const MatrixXs& G,
    const MatrixXs& H, const VectorXs& f, const VectorXs& q, const VectorXs& r,
    const VectorXs& g, const VectorXs& h) {
  const std::size_t nq = state_->get_nq();
  if (static_cast<std::size_t>(Aq.rows()) != nq) {
    throw_pretty(
        "Invalid argument: " << "Aq should be a squared matrix with size " +
                                    std::to_string(nq));
  }
  if (static_cast<std::size_t>(Av.rows()) != nq) {
    throw_pretty(
        "Invalid argument: " << "Av should be a squared matrix with size " +
                                    std::to_string(nq));
  }
  if (static_cast<std::size_t>(B.rows()) != nq) {
    throw_pretty(
        "Invalid argument: " << "B has wrong dimension (it should have " +
                                    std::to_string(nq) + " rows)");
  }
  if (static_cast<std::size_t>(Q.rows()) != 2 * nq ||
      static_cast<std::size_t>(Q.cols()) != 2 * nq) {
    throw_pretty(
        "Invalid argument: " << "Q has wrong dimension (it should be " +
                                    std::to_string(2 * nq) + " x " +
                                    std::to_string(2 * nq) + ")");
  }
  if (static_cast<std::size_t>(R.rows()) != nu_ ||
      static_cast<std::size_t>(R.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "R has wrong dimension (it should be " +
                                    std::to_string(nu_) + " x " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(N.rows()) != 2 * nq ||
      static_cast<std::size_t>(N.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "N has wrong dimension (it should be " +
                                    std::to_string(2 * nq) + " x " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(G.rows()) != ng_ ||
      static_cast<std::size_t>(G.cols()) != 2 * nq + nu_) {
    throw_pretty(
        "Invalid argument: " << "G has wrong dimension (it should be " +
                                    std::to_string(ng_) + " x " +
                                    std::to_string(2 * nq + nu_) + ")");
  }
  if (static_cast<std::size_t>(H.rows()) != nh_ ||
      static_cast<std::size_t>(H.cols()) != 2 * nq + nu_) {
    throw_pretty(
        "Invalid argument: " << "H has wrong dimension (it should be " +
                                    std::to_string(nh_) + " x " +
                                    std::to_string(2 * nq + nu_) + ")");
  }
  if (static_cast<std::size_t>(f.size()) != nq) {
    throw_pretty(
        "Invalid argument: " << "f has wrong dimension (it should be " +
                                    std::to_string(nq) + ")");
  }
  if (static_cast<std::size_t>(q.size()) != 2 * nq) {
    throw_pretty(
        "Invalid argument: " << "q has wrong dimension (it should be " +
                                    std::to_string(2 * nq) + ")");
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
  L_ = MatrixXs::Zero(2 * nq + nu_, 2 * nq + nu_);
  L_ << Q, N, N.transpose(), R;
  if (!checkPSD(L_)) {
    throw_pretty("Invalid argument "
                 << "[Q, N; N.T, R] is not positive semi-definite");
  }

  Aq_ = Aq;
  Av_ = Av;
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
