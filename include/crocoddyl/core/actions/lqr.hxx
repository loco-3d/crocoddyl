///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
LQRParamsTpl<Scalar>::LQRParamsTpl(std::shared_ptr<StateAbstract> state,
                                   const std::size_t np)
    : Base(state, np) {}

template <typename Scalar>
LQRParamsTpl<Scalar>::LQRParamsTpl(const std::size_t nx, const std::size_t np)
    : Base(std::make_shared<StateVector>(nx), np) {}

template <typename Scalar>
void LQRParamsTpl<Scalar>::update(
    const std::shared_ptr<ParamsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != this->np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(this->np_) + ")");
  }
  data->p = p;
}

template <typename Scalar>
void LQRParamsTpl<Scalar>::computeParamSensitivity(
    const std::shared_ptr<ActionDataAbstract>&,
    const std::shared_ptr<ParamsDataAbstract>& params,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  params->dx_dp.setZero();
}

template <typename Scalar>
template <typename NewScalar>
LQRParamsTpl<NewScalar> LQRParamsTpl<Scalar>::cast() const {
  typedef LQRParamsTpl<NewScalar> ReturnType;
  ReturnType ret(std::static_pointer_cast<StateAbstractTpl<NewScalar>>(
                     this->state_->template cast<NewScalar>()),
                 this->np_);
  ret.set_lb(this->lb_.template cast<NewScalar>());
  ret.set_ub(this->ub_.template cast<NewScalar>());
  return ret;
}

template <typename Scalar>
void LQRParamsTpl<Scalar>::print(std::ostream& os) const {
  os << "LQRParams {np=" << this->np_ << "}";
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const MatrixXs& A,
                                             const MatrixXs& B,
                                             const MatrixXs& Q,
                                             const MatrixXs& R,
                                             const MatrixXs& N)
    : Base(std::make_shared<StateVector>(A.cols()), B.cols(), 0),
      drift_free_(true),
      params_(nullptr) {
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
      params_(nullptr) {
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
      params_(nullptr) {
  set_LQR(A, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(
    const MatrixXs& A, const MatrixXs& B, const MatrixXs& P, const MatrixXs& Q,
    const MatrixXs& R, const MatrixXs& N, const MatrixXs& W, const MatrixXs& Y,
    const MatrixXs& V, const MatrixXs& G, const MatrixXs& H, const VectorXs& f,
    const VectorXs& q, const VectorXs& r, const VectorXs& m, const VectorXs& g,
    const VectorXs& h)
    : Base(std::make_shared<StateVector>(A.cols()), B.cols(), 0, G.rows(),
           H.rows(), G.rows(), H.rows(), P.cols()),
      drift_free_(false),
      params_(nullptr) {
  set_LQR(A, B, P, Q, R, N, W, Y, V, G, H, f, q, r, m, g, h);
}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const std::size_t nx,
                                             const std::size_t nu,
                                             const bool drift_free)
    : ActionModelLQRTpl(nx, nu, 0, 0, 0, drift_free) {}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(
    const std::size_t nx, const std::size_t nu, const std::size_t np,
    const std::size_t ng, const std::size_t nh, const bool drift_free)
    : Base(std::make_shared<StateVector>(nx), nu, 0, ng, nh, ng, nh, np),
      A_(MatrixXs::Identity(nx, nx)),
      B_(MatrixXs::Identity(nx, nu)),
      P_(MatrixXs::Identity(nx, np)),
      Q_(MatrixXs::Identity(nx, nx)),
      R_(MatrixXs::Identity(nu, nu)),
      N_(MatrixXs::Zero(nx, nu)),
      W_(MatrixXs::Identity(np, np)),
      Y_(MatrixXs::Zero(nx, np)),
      V_(MatrixXs::Zero(nu, np)),
      G_(MatrixXs::Identity(ng, nx + nu + np)),
      H_(MatrixXs::Identity(nh, nx + nu + np)),
      f_(VectorXs::Constant(nx, drift_free ? Scalar(0) : Scalar(1))),
      q_(VectorXs::Ones(nx)),
      r_(VectorXs::Ones(nu)),
      m_(VectorXs::Ones(np)),
      g_(VectorXs::Zero(ng)),
      h_(VectorXs::Zero(nh)),
      drift_free_(drift_free),
      params_(nullptr) {}

template <typename Scalar>
ActionModelLQRTpl<Scalar>::ActionModelLQRTpl(const ActionModelLQRTpl& copy)
    : Base(std::make_shared<StateVector>(copy.get_A().cols()),
           copy.get_B().cols(), 0, copy.get_G().rows(), copy.get_H().rows(),
           copy.get_G().rows(), copy.get_H().rows(), copy.get_P().cols()),
      drift_free_(copy.drift_free_),
      params_(nullptr) {
  set_LQR(copy.get_A(), copy.get_B(), copy.get_P(), copy.get_Q(), copy.get_R(),
          copy.get_N(), copy.get_W(), copy.get_Y(), copy.get_V(), copy.get_G(),
          copy.get_H(), copy.get_f(), copy.get_q(), copy.get_r(), copy.get_m(),
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
  const VectorXs& p = d->params != nullptr ? d->params->params->p : d->p;

  data->xnext.noalias() = A_ * x;
  data->xnext.noalias() += B_ * u;
  data->xnext.noalias() += P_ * p;
  data->xnext += f_;

  // cost = 0.5 * x^T * Q * x + 0.5 * u^T * R * u + x^T * N * u + q^T * x + r^T
  // * u
  d->Q_x_tmp.noalias() = Q_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Q_x_tmp);
  d->R_u_tmp.noalias() = R_ * u;
  data->cost += Scalar(0.5) * u.dot(d->R_u_tmp);
  d->Q_x_tmp.noalias() = N_ * u;
  data->cost += x.dot(d->Q_x_tmp);
  d->W_p_tmp.noalias() = W_ * p;
  data->cost += Scalar(0.5) * p.dot(d->W_p_tmp);
  d->Q_x_tmp.noalias() = Y_ * p;
  data->cost += x.dot(d->Q_x_tmp);
  d->R_u_tmp.noalias() = V_ * p;
  data->cost += u.dot(d->R_u_tmp);
  data->cost += q_.dot(x);
  data->cost += r_.dot(u);
  data->cost += m_.dot(p);

  // constraints
  const std::size_t nx = state_->get_nx();
  data->g.noalias() = G_.leftCols(nx) * x;
  data->g.noalias() += G_.middleCols(nx, nu_) * u;
  data->g.noalias() += G_.rightCols(np_) * p;
  data->g += g_;
  data->h.noalias() = H_.leftCols(nx) * x;
  data->h.noalias() += H_.middleCols(nx, nu_) * u;
  data->h.noalias() += H_.rightCols(np_) * p;
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
  const VectorXs& p = d->params != nullptr ? d->params->params->p : d->p;

  d->xnext = x;

  // cost = 0.5 * x^T * Q * x + q^T * x
  d->Q_x_tmp.noalias() = Q_ * x;
  data->cost = Scalar(0.5) * x.dot(d->Q_x_tmp);
  d->W_p_tmp.noalias() = W_ * p;
  data->cost += Scalar(0.5) * p.dot(d->W_p_tmp);
  d->Q_x_tmp.noalias() = Y_ * p;
  data->cost += x.dot(d->Q_x_tmp);
  data->cost += q_.dot(x);
  data->cost += m_.dot(p);

  // constraints
  const std::size_t nx = state_->get_nx();
  data->g.noalias() = G_.leftCols(nx) * x;
  data->g.noalias() += G_.rightCols(np_) * p;
  data->g += g_;
  data->h.noalias() = H_.leftCols(nx) * x;
  data->h.noalias() += H_.rightCols(np_) * p;
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
  Data* d = static_cast<Data*>(data.get());
  const VectorXs& p = d->params != nullptr ? d->params->params->p : d->p;
  data->Fx = A_;
  data->Fu = B_;
  data->Fp = P_;
  data->Lxx = Q_;
  data->Luu = R_;
  data->Lxu = N_;
  data->Lpp = W_;
  data->Lpx = Y_.transpose();
  data->Lpu = V_.transpose();
  data->Gx = G_.leftCols(nx);
  data->Gu = G_.middleCols(nx, nu_);
  data->Gp = G_.rightCols(np_);
  data->Hx = H_.leftCols(nx);
  data->Hu = H_.middleCols(nx, nu_);
  data->Hp = H_.rightCols(np_);
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
  data->Lx.noalias() += N_ * u;
  data->Lx.noalias() += Y_ * p;
  data->Lu = r_;
  data->Lu.noalias() += N_.transpose() * x;
  data->Lu.noalias() += R_ * u;
  data->Lu.noalias() += V_ * p;
  data->Lp = m_;
  data->Lp.noalias() += Y_.transpose() * x;
  data->Lp.noalias() += V_.transpose() * u;
  data->Lp.noalias() += W_ * p;
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
  Data* d = static_cast<Data*>(data.get());
  const VectorXs& p = d->params != nullptr ? d->params->params->p : d->p;
  data->Lxx = Q_;
  data->Lpp = W_;
  data->Lpx = Y_.transpose();
  data->Gx = G_.leftCols(nx);
  data->Gp = G_.rightCols(np_);
  data->Hx = H_.leftCols(nx);
  data->Hp = H_.rightCols(np_);
  data->Lx = q_;
  data->Lx.noalias() += Q_ * x;
  data->Lx.noalias() += Y_ * p;
  data->Lp = m_;
  data->Lp.noalias() += Y_.transpose() * x;
  data->Lp.noalias() += W_ * p;
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelLQRTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManager>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelLQRTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>& params_data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    params_data);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (params == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (params->get_np() != np_) {
    throw_pretty("Invalid argument: params has wrong dimension (it should be " +
                 std::to_string(np_) + ")");
  }
  params_ = params;
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  d->params = d->params != nullptr ? d->params : params_->createData();
  update_p(data, params_->zero());
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
  d->p = p;
  if (params_ != nullptr) {
    if (d->params == nullptr) {
      throw_pretty("Invalid argument: parameter data is null");
    }
    params_->update(d->params, p);
    d->p = d->params->params->p;
  }
}

template <typename Scalar>
template <typename NewScalar>
ActionModelLQRTpl<NewScalar> ActionModelLQRTpl<Scalar>::cast() const {
  typedef ActionModelLQRTpl<NewScalar> ReturnType;
  ReturnType ret(A_.template cast<NewScalar>(), B_.template cast<NewScalar>(),
                 P_.template cast<NewScalar>(), Q_.template cast<NewScalar>(),
                 R_.template cast<NewScalar>(), N_.template cast<NewScalar>(),
                 W_.template cast<NewScalar>(), Y_.template cast<NewScalar>(),
                 V_.template cast<NewScalar>(), G_.template cast<NewScalar>(),
                 H_.template cast<NewScalar>(), f_.template cast<NewScalar>(),
                 q_.template cast<NewScalar>(), r_.template cast<NewScalar>(),
                 m_.template cast<NewScalar>(), g_.template cast<NewScalar>(),
                 h_.template cast<NewScalar>());
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
  MatrixXs A = matrix_random_cast<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                                  Eigen::Dynamic, Eigen::Dynamic>(nx, nx);
  MatrixXs B = matrix_random_cast<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                                  Eigen::Dynamic, Eigen::Dynamic>(nx, nu);
  MatrixXs L_tmp =
      matrix_random_cast<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                         Eigen::Dynamic, Eigen::Dynamic>(nx + nu, nx + nu);
  MatrixXs L = L_tmp.transpose() * L_tmp;
  const Eigen::Block<MatrixXs> Q = L.topLeftCorner(nx, nx);
  const Eigen::Block<MatrixXs> R = L.bottomRightCorner(nu, nu);
  const Eigen::Block<MatrixXs> N = L.topRightCorner(nx, nu);
  MatrixXs G = matrix_random_cast<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                                  Eigen::Dynamic, Eigen::Dynamic>(ng, nx + nu);
  MatrixXs H = matrix_random_cast<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0,
                                  Eigen::Dynamic, Eigen::Dynamic>(nh, nx + nu);
  VectorXs f =
      vector_random_cast<Scalar, Eigen::Dynamic, 0, Eigen::Dynamic>(nx);
  VectorXs q =
      vector_random_cast<Scalar, Eigen::Dynamic, 0, Eigen::Dynamic>(nx);
  VectorXs r =
      vector_random_cast<Scalar, Eigen::Dynamic, 0, Eigen::Dynamic>(nu);
  VectorXs g =
      vector_random_cast<Scalar, Eigen::Dynamic, 0, Eigen::Dynamic>(ng);
  VectorXs h =
      vector_random_cast<Scalar, Eigen::Dynamic, 0, Eigen::Dynamic>(nh);
  return ActionModelLQRTpl<Scalar>(A, B, Q, R, N, G, H, f, q, r, g, h);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::print(std::ostream& os) const {
  os << "ActionModelLQR {nx=" << state_->get_nx() << ", nu=" << nu_
     << ", np=" << np_ << ", ng=" << ng_ << ", nh=" << nh_
     << ", drift_free=" << drift_free_ << "}";
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
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_P()
    const {
  return P_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_W()
    const {
  return W_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_Y()
    const {
  return Y_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& ActionModelLQRTpl<Scalar>::get_V()
    const {
  return V_;
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
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelLQRTpl<Scalar>::get_m()
    const {
  return m_;
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
void ActionModelLQRTpl<Scalar>::set_A(const MatrixXs& A) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(A.rows()) != nx ||
      static_cast<std::size_t>(A.cols()) != nx) {
    throw_pretty(
        "Invalid argument: " << "A should be a squared matrix with size " +
                                    std::to_string(nx));
  }
  A_ = A;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_B(const MatrixXs& B) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(B.rows()) != nx ||
      static_cast<std::size_t>(B.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "B has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(nu_) + ")");
  }
  B_ = B;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_P(const MatrixXs& P) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(P.rows()) != nx ||
      static_cast<std::size_t>(P.cols()) != np_) {
    throw_pretty("Invalid argument: "
                 << "P has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(np_) + ")");
  }
  P_ = P;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Q(const MatrixXs& Q) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(Q.rows()) != nx ||
      static_cast<std::size_t>(Q.cols()) != nx) {
    throw_pretty("Invalid argument: "
                 << "Q has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(nx) + ")");
  }
  Q_ = Q;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_R(const MatrixXs& R) {
  if (static_cast<std::size_t>(R.rows()) != nu_ ||
      static_cast<std::size_t>(R.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "R has wrong dimension (it should be " +
                                    std::to_string(nu_) + " x " +
                                    std::to_string(nu_) + ")");
  }
  R_ = R;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_N(const MatrixXs& N) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(N.rows()) != nx ||
      static_cast<std::size_t>(N.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "N has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(nu_) + ")");
  }
  N_ = N;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_W(const MatrixXs& W) {
  if (static_cast<std::size_t>(W.rows()) != np_ ||
      static_cast<std::size_t>(W.cols()) != np_) {
    throw_pretty(
        "Invalid argument: " << "W has wrong dimension (it should be " +
                                    std::to_string(np_) + " x " +
                                    std::to_string(np_) + ")");
  }
  W_ = W;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_Y(const MatrixXs& Y) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(Y.rows()) != nx ||
      static_cast<std::size_t>(Y.cols()) != np_) {
    throw_pretty("Invalid argument: "
                 << "Y has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(np_) + ")");
  }
  Y_ = Y;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_V(const MatrixXs& V) {
  if (static_cast<std::size_t>(V.rows()) != nu_ ||
      static_cast<std::size_t>(V.cols()) != np_) {
    throw_pretty(
        "Invalid argument: " << "V has wrong dimension (it should be " +
                                    std::to_string(nu_) + " x " +
                                    std::to_string(np_) + ")");
  }
  V_ = V;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_G(const MatrixXs& G) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(G.rows()) != ng_ ||
      static_cast<std::size_t>(G.cols()) != nx + nu_ + np_) {
    throw_pretty(
        "Invalid argument: " << "G has wrong dimension (it should be " +
                                    std::to_string(ng_) + " x " +
                                    std::to_string(nx + nu_ + np_) + ")");
  }
  G_ = G;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_H(const MatrixXs& H) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(H.rows()) != nh_ ||
      static_cast<std::size_t>(H.cols()) != nx + nu_ + np_) {
    throw_pretty(
        "Invalid argument: " << "H has wrong dimension (it should be " +
                                    std::to_string(nh_) + " x " +
                                    std::to_string(nx + nu_ + np_) + ")");
  }
  H_ = H;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_f(const VectorXs& f) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(f.size()) != nx) {
    throw_pretty(
        "Invalid argument: " << "f has wrong dimension (it should be " +
                                    std::to_string(nx) + ")");
  }
  f_ = f;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_q(const VectorXs& q) {
  const std::size_t nx = state_->get_nx();
  if (static_cast<std::size_t>(q.size()) != nx) {
    throw_pretty(
        "Invalid argument: " << "q has wrong dimension (it should be " +
                                    std::to_string(nx) + ")");
  }
  q_ = q;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_r(const VectorXs& r) {
  if (static_cast<std::size_t>(r.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "r has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  r_ = r;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_m(const VectorXs& m) {
  if (static_cast<std::size_t>(m.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "m has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  m_ = m;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_g(const VectorXs& g) {
  if (static_cast<std::size_t>(g.size()) != ng_) {
    throw_pretty(
        "Invalid argument: " << "g has wrong dimension (it should be " +
                                    std::to_string(ng_) + ")");
  }
  g_ = g;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_h(const VectorXs& h) {
  if (static_cast<std::size_t>(h.size()) != nh_) {
    throw_pretty(
        "Invalid argument: " << "h has wrong dimension (it should be " +
                                    std::to_string(nh_) + ")");
  }
  h_ = h;
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_LQR(const MatrixXs& A, const MatrixXs& B,
                                        const MatrixXs& Q, const MatrixXs& R,
                                        const MatrixXs& N, const MatrixXs& G,
                                        const MatrixXs& H, const VectorXs& f,
                                        const VectorXs& q, const VectorXs& r,
                                        const VectorXs& g, const VectorXs& h) {
  const std::size_t nx = state_->get_nx();
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
  MatrixXs P = MatrixXs::Zero(nx, np_);
  MatrixXs W = MatrixXs::Zero(np_, np_);
  MatrixXs Y = MatrixXs::Zero(nx, np_);
  MatrixXs V = MatrixXs::Zero(nu_, np_);
  MatrixXs G_param = MatrixXs::Zero(ng_, nx + nu_ + np_);
  MatrixXs H_param = MatrixXs::Zero(nh_, nx + nu_ + np_);
  G_param.leftCols(nx + nu_) = G;
  H_param.leftCols(nx + nu_) = H;
  VectorXs m = VectorXs::Zero(np_);
  set_LQR(A, B, P, Q, R, N, W, Y, V, G_param, H_param, f, q, r, m, g, h);
}

template <typename Scalar>
void ActionModelLQRTpl<Scalar>::set_LQR(
    const MatrixXs& A, const MatrixXs& B, const MatrixXs& P, const MatrixXs& Q,
    const MatrixXs& R, const MatrixXs& N, const MatrixXs& W, const MatrixXs& Y,
    const MatrixXs& V, const MatrixXs& G, const MatrixXs& H, const VectorXs& f,
    const VectorXs& q, const VectorXs& r, const VectorXs& m, const VectorXs& g,
    const VectorXs& h) {
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
  if (static_cast<std::size_t>(B.cols()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "B has wrong dimension (it should have " +
                                    std::to_string(nu_) + " columns)");
  }
  if (static_cast<std::size_t>(P.rows()) != nx ||
      static_cast<std::size_t>(P.cols()) != np_) {
    throw_pretty("Invalid argument: "
                 << "P has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(np_) + ")");
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
  if (static_cast<std::size_t>(W.rows()) != np_ ||
      static_cast<std::size_t>(W.cols()) != np_) {
    throw_pretty(
        "Invalid argument: " << "W has wrong dimension (it should be " +
                                    std::to_string(np_) + " x " +
                                    std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(Y.rows()) != nx ||
      static_cast<std::size_t>(Y.cols()) != np_) {
    throw_pretty("Invalid argument: "
                 << "Y has wrong dimension (it should be " +
                        std::to_string(nx) + " x " + std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(V.rows()) != nu_ ||
      static_cast<std::size_t>(V.cols()) != np_) {
    throw_pretty(
        "Invalid argument: " << "V has wrong dimension (it should be " +
                                    std::to_string(nu_) + " x " +
                                    std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(G.rows()) != ng_ ||
      static_cast<std::size_t>(G.cols()) != nx + nu_ + np_) {
    throw_pretty(
        "Invalid argument: " << "G has wrong dimension (it should be " +
                                    std::to_string(ng_) + " x " +
                                    std::to_string(nx + nu_ + np_) + ")");
  }
  if (static_cast<std::size_t>(H.rows()) != nh_ ||
      static_cast<std::size_t>(H.cols()) != nx + nu_ + np_) {
    throw_pretty(
        "Invalid argument: " << "H has wrong dimension (it should be " +
                                    std::to_string(nh_) + " x " +
                                    std::to_string(nx + nu_ + np_) + ")");
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
  if (static_cast<std::size_t>(m.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "m has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
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
  L_ = MatrixXs::Zero(nx + nu_ + np_, nx + nu_ + np_);
  L_.topLeftCorner(nx, nx) = Q;
  L_.block(0, nx, nx, nu_) = N;
  L_.block(nx, 0, nu_, nx) = N.transpose();
  L_.block(nx, nx, nu_, nu_) = R;
  if (np_ != 0) {
    L_.block(0, nx + nu_, nx, np_) = Y;
    L_.block(nx + nu_, 0, np_, nx) = Y.transpose();
    L_.block(nx, nx + nu_, nu_, np_) = V;
    L_.block(nx + nu_, nx, np_, nu_) = V.transpose();
    L_.bottomRightCorner(np_, np_) = W;
  }
  if (!checkPSD(L_)) {
    throw_pretty("Invalid argument "
                 << "[Q, N, Y; N.T, R, V; Y.T, V.T, W] is not positive "
                    "semi-definite");
  }
  A_ = A;
  B_ = B;
  P_ = P;
  f_ = f;
  Q_ = Q;
  R_ = R;
  N_ = N;
  W_ = W;
  Y_ = Y;
  V_ = V;
  G_ = G;
  H_ = H;
  q_ = q;
  r_ = r;
  m_ = m;
  g_ = g;
  h_ = h;
}

}  // namespace crocoddyl
