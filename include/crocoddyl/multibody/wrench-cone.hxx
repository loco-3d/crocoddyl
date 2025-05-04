///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, University of Oxford,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const Matrix3s& R, const Scalar mu,
                                     const Vector2s& box, const std::size_t nf,
                                     const bool inner_appr,
                                     const Scalar min_nforce,
                                     const Scalar max_nforce)
    : nf_(nf),
      R_(R),
      box_(box),
      mu_(mu),
      inner_appr_(inner_appr),
      min_nforce_(min_nforce),
      max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1."
              << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0"
              << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: max_nforce has to be a positive value, set to "
                 "infinity value"
              << std::endl;
  }
  A_ = MatrixX6s::Zero(nf_ + 13, 6);
  ub_ = VectorXs::Zero(nf_ + 13);
  lb_ = VectorXs::Zero(nf_ + 13);
  center_ = Vector2s::Zero();

  // Update the inequality matrix and bounds
  update();
}

  template <typename Scalar>
  WrenchConeTpl<Scalar>::WrenchConeTpl(const Matrix3s& R, const Scalar mu,
                                       const Vector2s& box, const Vector2s& center,
                                       const std::size_t nf,
                                       const bool inner_appr,
                                       const Scalar min_nforce,
                                       const Scalar max_nforce)
      : nf_(nf),
        R_(R),
        box_(box),
        center_(center),
        mu_(mu),
        inner_appr_(inner_appr),
        min_nforce_(min_nforce),
        max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1."
              << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0"
              << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: max_nforce has to be a positive value, set to "
                 "infinity value"
              << std::endl;
  }
  A_ = MatrixX6s::Zero(nf_ + 13, 6);
  ub_ = VectorXs::Zero(nf_ + 13);
  lb_ = VectorXs::Zero(nf_ + 13);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const Matrix3s& R, const Scalar mu,
                                     const Vector2s& box, std::size_t nf,
                                     const Scalar min_nforce,
                                     const Scalar max_nforce)
    : nf_(nf),
      R_(R),
      box_(box),
      mu_(mu),
      inner_appr_(true),
      min_nforce_(min_nforce),
      max_nforce_(max_nforce) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1."
              << std::endl;
  }
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0"
              << std::endl;
  }
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: max_nforce has to be a positive value, set to "
                 "infinity value"
              << std::endl;
  }
  A_ = MatrixX6s::Zero(nf_ + 13, 6);
  ub_ = VectorXs::Zero(nf_ + 13);
  lb_ = VectorXs::Zero(nf_ + 13);
  center_ = Vector2s::Zero();

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl(const WrenchConeTpl<Scalar>& cone)
    : nf_(cone.get_nf()),
      A_(cone.get_A()),
      ub_(cone.get_ub()),
      lb_(cone.get_lb()),
      R_(cone.get_R()),
      box_(cone.get_box()),
      center_(cone.get_center()),
      mu_(cone.get_mu()),
      inner_appr_(cone.get_inner_appr()),
      min_nforce_(cone.get_min_nforce()),
      max_nforce_(cone.get_max_nforce()) {}

template <typename Scalar>
WrenchConeTpl<Scalar>::WrenchConeTpl()
    : nf_(4),
      A_(nf_ + 13, 6),
      ub_(nf_ + 13),
      lb_(nf_ + 13),
      R_(Matrix3s::Identity()),
      box_(std::numeric_limits<Scalar>::infinity(),
           std::numeric_limits<Scalar>::infinity()),
      center_(Vector2s::Zero()),
      mu_(Scalar(0.7)),
      inner_appr_(true),
      min_nforce_(Scalar(0.)),
      max_nforce_(std::numeric_limits<Scalar>::infinity()) {
  A_.setZero();
  ub_.setZero();
  lb_.setZero();

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
WrenchConeTpl<Scalar>::~WrenchConeTpl() {}

template <typename Scalar>
void WrenchConeTpl<Scalar>::update() {
  // Initialize the matrix and bounds
  A_.setZero();
  ub_.setZero();
  lb_.setOnes();
  lb_ *= -std::numeric_limits<Scalar>::infinity();

  // Compute the mu given the type of friction cone approximation
  Scalar mu = mu_;
  Scalar theta =
      static_cast<Scalar>(2.0) * pi<Scalar>() / static_cast<Scalar>(nf_);
  if (inner_appr_) {
    mu *= cos(theta * Scalar(0.5));
  }

  // Create a temporary object for computation
  Matrix3s R_transpose = R_.transpose();
  // There are blocks of the A matrix that are repeated. By separating it this way, we can
  // reduced computation.
  MatrixX3s A_left = A_.template leftCols<nf_>();
  MatrixX3s A_right = A_.template rightCols<nf_>() ;

  // Friction cone information
  // This segment of matrix is defined as
  // [ 1  0 -mu  0  0  0;     fx <= mu fz  = fx_max
  //  -1  0 -mu  0  0  0;     fx >= -mu fz = fx_min
  //   0  1 -mu  0  0  0;     fy <= mu fz  = fy_max
  //   0 -1 -mu  0  0  0;     fy >= -mu fz = fy_min
  //   0  0   1  0  0  0]     fz >= 0      = fz_min
  Vector3s mu_nsurf = -mu * Vector3s::UnitZ(); // We can pull this out because it's reused.
  std::size_t row = 0;
  for (std::size_t i = 0; i < nf_ / 2; ++i) {
    Scalar theta_i = theta * static_cast<Scalar>(i);
    Vector3s tsurf_i = Vector3s(cos(theta_i), sin(theta_i), Scalar(0.));
    A_.row(row++).template head<3>() =
        (mu_nsurf + tsurf_i).transpose() * R_transpose;
    A_.row(row++).template head<3>() =
        (mu_nsurf - tsurf_i).transpose() * R_transpose;
  }
  A_.row(nf_).template head<3>() = R_transpose.row(2);
  lb_(nf_) = min_nforce_;
  ub_(nf_) = max_nforce_;

  // Box dimensions
  const Scalar L = box_(0) * Scalar(0.5);
  const Scalar W = box_(1) * Scalar(0.5);
  const Scalar x = center_(0);
  const Scalar y = center_(1);
  const Scalar left_bound =   W + y;
  const Scalar right_bound = -W + y;
  const Scalar front_bound =  L + x;
  const Scalar back_bound =  -L + x;

  // CoP information
  // This segment of matrix is defined as
  // [0  0 -W  1  0  0;   -W fz + tau_x <= 0 ->  tau_x <= W fz,                   = tau_x_max, so W is the left_bound
  //  0  0 -W -1  0  0;   -W fz - tau_x <= 0 -> -tau_x <= W fz -> tau_x >= -W fz, = tau_x_min, so -W is the right_bound
  //  0  0 -L  0  1  0;   -L fz + tau_y <= 0 ->  tau_y <= L fz,                   = tau_y_max, so L is -back_bound
  //  0  0 -L  0 -1  0]   -L fz - tau_y <= 0 -> -tau_y <= L fz -> tau_y >= -L fz, = tau_y_min, so L is front_bound
  A_.row(nf_ + 1) << -left_bound  * R_transpose.row(2),  R_transpose.row(0);
  A_.row(nf_ + 2) <<  right_bound * R_transpose.row(2), -R_transepose.row(0);
  A_.row(nf_ + 3) <<  back_bound  * R_transpose.row(2),  R_transpose.row(1);
  A_.row(nf_ + 4) << -front_bound * R_transpose.row(2), -R_transpose.row(1);

  // Yaw-tau information
  // Here, we use the transformation between the sole frame (0, 0) and the center of this polygon r = (x, y). The CWC
  // constraints are formulated for this polygon, which has a torque tau_hat. We know then that
  // tau_hat = tau - center X f.
  const Scalar mu_LW = -mu * (L + W);
  // The segment of the matrix that encodes the minimum torque is defined as
  // [ W  L -mu*(L+W) -mu -mu -1;   ->  (W-y)  (L+x) -mu*(L+W+x-y) -mu -mu -1;
  //   W -L -mu*(L+W) -mu  mu -1;   ->  (W-y) -(L-x) -mu*(L+W-x-y) -mu  mu -1;
  //  -W  L -mu*(L+W)  mu -mu -1;   -> -(W+y)  (L+x) -mu*(L+W+x+y)  mu -mu -1;
  //  -W -L -mu*(L+W)  mu  mu -1]   -> -(W+y) -(L-x) -mu*(L+W-x+y)  mu  mu -1]
  A_.row(nf_ + 5) << Vector3s(W - y, L + x, mu_LW - mu * (x - y)).transpose() * R_transpose,
      Vector3s(-mu, -mu, Scalar(-1.)).transpose() * R_transpose;
  A_.row(nf_ + 6) << Vector3s(W - y, -(L - x), mu_LW + mu * (x + y)).transpose() * R_transpose,
      Vector3s(-mu, mu, Scalar(-1.)).transpose() * R_transpose;
  A_.row(nf_ + 7) << Vector3s(-(W + y), L + x, mu_LW - mu * (x + y)).transpose() * R_transpose,
      Vector3s(mu, -mu, Scalar(-1.)).transpose() * R_transpose;
  A_.row(nf_ + 8) << Vector3s(-(W + y), -(L - x), mu_LW - mu * (y - x)).transpose() * R_transpose,
      Vector3s(mu, mu, Scalar(-1.)).transpose() * R_transpose;
  // The segment of the matrix that encodes the infinity torque is defined as
  // [ W  L -mu*(L+W)  mu  mu 1;    ->  (W+y)  (L-x) -mu*(L+W-x+y)  mu  mu 1;
  //   W -L -mu*(L+W)  mu -mu 1;    ->  (W+y) -(L+x) -mu*(L+W+x+y)  mu -mu 1;
  //  -W  L -mu*(L+W) -mu  mu 1;    -> -(W-y)  (L-x) -mu*(L+W-x-y) -mu  mu 1;
  //  -W -L -mu*(L+W) -mu -mu 1]    -> -(W-y) -(L+x) -mu*(L+W+x-y) -mu -mu 1]
  A_.row(nf_ + 9) << Vector3s(W + y, L - x, mu_LW - mu * (y - x)).transpose() * R_transpose,
      Vector3s(mu, mu, Scalar(1.)).transpose() * R_transpose;
  A_.row(nf_ + 10) << Vector3s(W + y, -(L + x), mu_LW - mu * (x + y)).transpose() * R_transpose,
      Vector3s(mu, -mu, Scalar(1.)).transpose() * R_transpose;
  A_.row(nf_ + 11) << Vector3s(-(W - y), L - x, mu_LW + mu * (x + y)).transpose() * R_transpose,
      Vector3s(-mu, mu, Scalar(1.)).transpose() * R_transpose;
  A_.row(nf_ + 12) << Vector3s(-(W - y), -(L + x), mu_LW - mu * (x - y)).transpose() * R_transpose,
      Vector3s(-mu, -mu, Scalar(1.)).transpose() * R_transpose;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::update(const Matrix3s& R, const Scalar mu,
                                   const Vector2s& box, const Scalar min_nforce,
                                   const Scalar max_nforce) {
  set_R(R);
  set_mu(mu);
  set_inner_appr(inner_appr_);
  set_box(box);
  set_min_nforce(min_nforce);
  set_max_nforce(max_nforce);

  // Update the inequality matrix and bounds
  update();
}

template <typename Scalar>
template <typename NewScalar>
WrenchConeTpl<NewScalar> WrenchConeTpl<Scalar>::cast() const {
  typedef WrenchConeTpl<NewScalar> ReturnType;
  ReturnType ret(R_.template cast<NewScalar>(), scalar_cast<NewScalar>(mu_),
                 box_.template cast<NewScalar>(), nf_, inner_appr_,
                 scalar_cast<NewScalar>(min_nforce_),
                 scalar_cast<NewScalar>(max_nforce_));
  return ret;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixX6s& WrenchConeTpl<Scalar>::get_A()
    const {
  return A_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& WrenchConeTpl<Scalar>::get_ub()
    const {
  return ub_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& WrenchConeTpl<Scalar>::get_lb()
    const {
  return lb_;
}

template <typename Scalar>
std::size_t WrenchConeTpl<Scalar>::get_nf() const {
  return nf_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s& WrenchConeTpl<Scalar>::get_R()
    const {
  return R_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s& WrenchConeTpl<Scalar>::get_box()
    const {
  return box_;
}

template <typename Scalar>
const Scalar WrenchConeTpl<Scalar>::get_mu() const {
  return mu_;
}

template <typename Scalar>
bool WrenchConeTpl<Scalar>::get_inner_appr() const {
  return inner_appr_;
}

template <typename Scalar>
const Scalar WrenchConeTpl<Scalar>::get_min_nforce() const {
  return min_nforce_;
}

template <typename Scalar>
const Scalar WrenchConeTpl<Scalar>::get_max_nforce() const {
  return max_nforce_;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_R(const Matrix3s& R) {
  R_ = R;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_box(const Vector2s& box) {
  box_ = box;
  if (box_(0) < Scalar(0.)) {
    box_(0) = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: box(0) has to be a positive value, set to max. float"
              << std::endl;
  }
  if (box_(1) < Scalar(0.)) {
    box_(1) = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: box(0) has to be a positive value, set to max. float"
              << std::endl;
  }
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_mu(const Scalar mu) {
  mu_ = mu;
  if (mu < Scalar(0.)) {
    mu_ = Scalar(1.);
    std::cerr << "Warning: mu has to be a positive value, set to 1."
              << std::endl;
  }
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_inner_appr(const bool inner_appr) {
  inner_appr_ = inner_appr;
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_min_nforce(const Scalar min_nforce) {
  min_nforce_ = min_nforce;
  if (min_nforce < Scalar(0.)) {
    min_nforce_ = Scalar(0.);
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0"
              << std::endl;
  }
}

template <typename Scalar>
void WrenchConeTpl<Scalar>::set_max_nforce(const Scalar max_nforce) {
  max_nforce_ = max_nforce;
  if (max_nforce < Scalar(0.)) {
    max_nforce_ = std::numeric_limits<Scalar>::infinity();
    std::cerr << "Warning: max_nforce has to be a positive value, set to "
                 "infinity value"
              << std::endl;
  }
}

template <typename Scalar>
WrenchConeTpl<Scalar>& WrenchConeTpl<Scalar>::operator=(
    const WrenchConeTpl<Scalar>& other) {
  if (this != &other) {
    nf_ = other.get_nf();
    A_ = other.get_A();
    ub_ = other.get_ub();
    lb_ = other.get_lb();
    R_ = other.get_R();
    box_ = other.get_box();
    mu_ = other.get_mu();
    inner_appr_ = other.get_inner_appr();
    min_nforce_ = other.get_min_nforce();
    max_nforce_ = other.get_max_nforce();
  }
  return *this;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const WrenchConeTpl<Scalar>& X) {
  os << "         R: " << X.get_R() << std::endl;
  os << "        mu: " << X.get_mu() << std::endl;
  os << "       box: " << X.get_box().transpose() << std::endl;
  os << "        nf: " << X.get_nf() << std::endl;
  os << "inner_appr: ";
  if (X.get_inner_appr()) {
    os << "true" << std::endl;
  } else {
    os << "false" << std::endl;
  }
  os << " min_force: " << X.get_min_nforce() << std::endl;
  os << " max_force: " << X.get_max_nforce() << std::endl;
  return os;
}

}  // namespace crocoddyl
