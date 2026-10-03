///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
LogCholeskyParametrizationDataTpl<Scalar>::LogCholeskyParametrizationDataTpl()
    : alpha(0.),
      d1(0.),
      d2(0.),
      d3(0.),
      s1(0.),
      s2(0.),
      s3(0.),
      t1(0.),
      t2(0.),
      t3(0.),
      exp2alpha(0.),
      exp2alpha2(0.),
      expd1(0.),
      expd2(0.),
      expd3(0.),
      exp2d1(0.),
      exp2d2(0.),
      exp2d3(0.),
      s1pow2(0.),
      s2pow2(0.),
      s3pow2(0.),
      t1pow2(0.),
      t2pow2(0.),
      t3pow2(0.) {}

template <typename Scalar>
std::shared_ptr<typename LogCholeskyParametrizationTpl<
    Scalar>::LogCholeskyParametrizationData>
LogCholeskyParametrizationTpl<Scalar>::castData(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data) const {
  std::shared_ptr<LogCholeskyParametrizationData> d =
      std::dynamic_pointer_cast<LogCholeskyParametrizationData>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data is not a LogCholeskyParametrizationData");
  }
  return d;
}

template <typename Scalar>
void LogCholeskyParametrizationTpl<Scalar>::fromParametrization(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data,
    Eigen::Ref<VectorXs> psi, const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != Base::kDimension ||
      static_cast<std::size_t>(psi.size()) != Base::kDimension) {
    throw_pretty("Invalid argument: p and psi must have dimension 10");
  }
  const std::shared_ptr<LogCholeskyParametrizationData> d = castData(data);

  d->alpha = p[0];
  d->d1 = p[1];
  d->d2 = p[2];
  d->d3 = p[3];
  d->s1 = p[4];
  d->s2 = p[5];
  d->s3 = p[6];
  d->t1 = p[7];
  d->t2 = p[8];
  d->t3 = p[9];
  using std::exp;
  d->exp2alpha = exp(Scalar(2.) * d->alpha);
  d->expd1 = exp(d->d1);
  d->expd2 = exp(d->d2);
  d->expd3 = exp(d->d3);
  d->exp2d1 = exp(Scalar(2.) * d->d1);
  d->exp2d2 = exp(Scalar(2.) * d->d2);
  d->exp2d3 = exp(Scalar(2.) * d->d3);
  d->s1pow2 = d->s1 * d->s1;
  d->s2pow2 = d->s2 * d->s2;
  d->s3pow2 = d->s3 * d->s3;

  psi[0] = d->exp2alpha *
           (d->t1 * d->t1 + d->t2 * d->t2 + d->t3 * d->t3 + Scalar(1.));
  psi[1] = d->exp2alpha * d->t1 * d->expd1;
  psi[2] = d->exp2alpha * (d->t1 * d->s1 + d->t2 * d->expd2);
  psi[3] = d->exp2alpha * (d->t1 * d->s3 + d->t2 * d->s2 + d->t3 * d->expd3);
  psi[4] = d->exp2alpha *
           (d->s1pow2 + d->s2pow2 + d->s3pow2 + d->exp2d2 + d->exp2d3);
  psi[5] = -d->exp2alpha * d->s1 * d->expd1;
  psi[6] = d->exp2alpha * (d->s2pow2 + d->s3pow2 + d->exp2d1 + d->exp2d3);
  psi[7] = -d->exp2alpha * d->s3 * d->expd1;
  psi[8] = -d->exp2alpha * (d->s1 * d->s3 + d->s2 * d->expd2);
  psi[9] = d->exp2alpha * (d->s1pow2 + d->exp2d1 + d->exp2d2);
}

template <typename Scalar>
void LogCholeskyParametrizationTpl<Scalar>::toParametrization(
    Eigen::Ref<VectorXs> p, const Eigen::Ref<const VectorXs>& psi) {
  if (static_cast<std::size_t>(p.size()) != Base::kDimension ||
      static_cast<std::size_t>(psi.size()) != Base::kDimension) {
    throw_pretty("Invalid argument: p and psi must have dimension 10");
  }

  using std::log;
  using std::sqrt;
  const Scalar exp_alpha_exp_d1 =
      sqrt(Scalar(0.5) * (psi[6] + psi[9] - psi[4]));
  const Scalar exp_alpha_s12 = -psi[5] / exp_alpha_exp_d1;
  const Scalar exp_alpha_s13 = -psi[7] / exp_alpha_exp_d1;
  const Scalar exp_alpha_exp_d2 =
      sqrt(psi[9] - exp_alpha_exp_d1 * exp_alpha_exp_d1 -
           exp_alpha_s12 * exp_alpha_s12);
  const Scalar exp_alpha_s23 =
      (-psi[8] - exp_alpha_s12 * exp_alpha_s13) / exp_alpha_exp_d2;
  const Scalar exp_alpha_exp_d3 =
      sqrt(psi[6] - exp_alpha_exp_d1 * exp_alpha_exp_d1 -
           exp_alpha_s13 * exp_alpha_s13 - exp_alpha_s23 * exp_alpha_s23);
  const Scalar exp_alpha_t1 = psi[1] / exp_alpha_exp_d1;
  const Scalar exp_alpha_t2 =
      (psi[2] - exp_alpha_t1 * exp_alpha_s12) / exp_alpha_exp_d2;
  const Scalar exp_alpha_t3 =
      (psi[3] - exp_alpha_t1 * exp_alpha_s13 - exp_alpha_t2 * exp_alpha_s23) /
      exp_alpha_exp_d3;
  const Scalar exp_alpha =
      sqrt(psi[0] - exp_alpha_t1 * exp_alpha_t1 - exp_alpha_t2 * exp_alpha_t2 -
           exp_alpha_t3 * exp_alpha_t3);

  p[0] = log(exp_alpha);
  p[1] = log(exp_alpha_exp_d1 / exp_alpha);
  p[2] = log(exp_alpha_exp_d2 / exp_alpha);
  p[3] = log(exp_alpha_exp_d3 / exp_alpha);
  p[4] = exp_alpha_s12 / exp_alpha;
  p[5] = exp_alpha_s23 / exp_alpha;
  p[6] = exp_alpha_s13 / exp_alpha;
  p[7] = exp_alpha_t1 / exp_alpha;
  p[8] = exp_alpha_t2 / exp_alpha;
  p[9] = exp_alpha_t3 / exp_alpha;
}

template <typename Scalar>
void LogCholeskyParametrizationTpl<Scalar>::updateParametrizationDerivative(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data,
    Eigen::Ref<MatrixXs> dpsi_dp, const Eigen::Ref<const VectorXs>& p,
    const Eigen::Ref<const VectorXs>& psi) {
  (void)psi;
  if (static_cast<std::size_t>(p.size()) != Base::kDimension ||
      static_cast<std::size_t>(dpsi_dp.rows()) != Base::kDimension ||
      static_cast<std::size_t>(dpsi_dp.cols()) != Base::kDimension) {
    throw_pretty("Invalid argument: p and dpsi_dp must have dimension 10");
  }
  const std::shared_ptr<LogCholeskyParametrizationData> d = castData(data);
  dpsi_dp.setZero();

  d->alpha = p[0];
  d->d1 = p[1];
  d->d2 = p[2];
  d->d3 = p[3];
  d->s1 = p[4];
  d->s2 = p[5];
  d->s3 = p[6];
  d->t1 = p[7];
  d->t2 = p[8];
  d->t3 = p[9];
  d->t1pow2 = d->t1 * d->t1;
  d->t2pow2 = d->t2 * d->t2;
  d->t3pow2 = d->t3 * d->t3;
  using std::exp;
  d->exp2alpha = exp(Scalar(2.) * d->alpha);
  d->exp2alpha2 = Scalar(2.) * d->exp2alpha;
  d->expd1 = exp(d->d1);
  d->expd2 = exp(d->d2);
  d->expd3 = exp(d->d3);
  d->exp2d1 = exp(Scalar(2.) * d->d1);
  d->exp2d2 = exp(Scalar(2.) * d->d2);
  d->exp2d3 = exp(Scalar(2.) * d->d3);
  d->s1pow2 = d->s1 * d->s1;
  d->s2pow2 = d->s2 * d->s2;
  d->s3pow2 = d->s3 * d->s3;

  dpsi_dp(0, 0) =
      d->exp2alpha2 * (d->t1pow2 + d->t2pow2 + d->t3pow2 + Scalar(1.));
  dpsi_dp(1, 0) = d->exp2alpha2 * d->t1 * d->expd1;
  dpsi_dp(2, 0) = d->exp2alpha2 * (d->t1 * d->s1 + d->t2 * d->expd2);
  dpsi_dp(3, 0) =
      d->exp2alpha2 * (d->t1 * d->s3 + d->t2 * d->s2 + d->t3 * d->expd3);
  dpsi_dp(4, 0) = d->exp2alpha2 *
                  (d->s1pow2 + d->s2pow2 + d->s3pow2 + d->exp2d2 + d->exp2d3);
  dpsi_dp(5, 0) = d->exp2alpha2 * (-d->s1 * d->expd1);
  dpsi_dp(6, 0) =
      d->exp2alpha2 * (d->s2pow2 + d->s3pow2 + d->exp2d1 + d->exp2d3);
  dpsi_dp(7, 0) = d->exp2alpha2 * (-d->s3 * d->expd1);
  dpsi_dp(8, 0) = d->exp2alpha2 * (-(d->s1 * d->s3) - (d->s2 * d->expd2));
  dpsi_dp(9, 0) = d->exp2alpha2 * (d->s1pow2 + d->exp2d1 + d->exp2d2);

  dpsi_dp(1, 1) = d->exp2alpha * d->t1 * d->expd1;
  dpsi_dp(5, 1) = -d->exp2alpha * d->s1 * d->expd1;
  dpsi_dp(6, 1) = d->exp2alpha2 * d->exp2d1;
  dpsi_dp(7, 1) = -d->exp2alpha * d->s3 * d->expd1;
  dpsi_dp(9, 1) = d->exp2alpha2 * d->exp2d1;

  dpsi_dp(2, 2) = d->exp2alpha * d->t2 * d->expd2;
  dpsi_dp(4, 2) = d->exp2alpha2 * d->exp2d2;
  dpsi_dp(8, 2) = -d->exp2alpha * d->s2 * d->expd2;
  dpsi_dp(9, 2) = d->exp2alpha2 * d->exp2d2;

  dpsi_dp(3, 3) = d->exp2alpha * d->t3 * d->expd3;
  dpsi_dp(4, 3) = d->exp2alpha2 * d->exp2d3;
  dpsi_dp(6, 3) = d->exp2alpha2 * d->exp2d3;

  dpsi_dp(2, 4) = d->exp2alpha * d->t1;
  dpsi_dp(4, 4) = d->exp2alpha2 * d->s1;
  dpsi_dp(5, 4) = -d->exp2alpha * d->expd1;
  dpsi_dp(8, 4) = -d->exp2alpha * d->s3;
  dpsi_dp(9, 4) = d->exp2alpha2 * d->s1;

  dpsi_dp(3, 5) = d->exp2alpha * d->t2;
  dpsi_dp(4, 5) = d->exp2alpha2 * d->s2;
  dpsi_dp(6, 5) = d->exp2alpha2 * d->s2;
  dpsi_dp(8, 5) = -d->exp2alpha * d->expd2;

  dpsi_dp(3, 6) = d->exp2alpha * d->t1;
  dpsi_dp(4, 6) = d->exp2alpha2 * d->s3;
  dpsi_dp(6, 6) = d->exp2alpha2 * d->s3;
  dpsi_dp(7, 6) = -d->exp2alpha * d->expd1;
  dpsi_dp(8, 6) = -d->exp2alpha * d->s1;

  dpsi_dp(0, 7) = d->exp2alpha * Scalar(2.) * d->t1;
  dpsi_dp(1, 7) = d->exp2alpha * d->expd1;
  dpsi_dp(2, 7) = d->exp2alpha * d->s1;
  dpsi_dp(3, 7) = d->exp2alpha * d->s3;

  dpsi_dp(0, 8) = d->exp2alpha * Scalar(2.) * d->t2;
  dpsi_dp(2, 8) = d->exp2alpha * d->expd2;
  dpsi_dp(3, 8) = d->exp2alpha * d->s2;

  dpsi_dp(0, 9) = d->exp2alpha * Scalar(2.) * d->t3;
  dpsi_dp(3, 9) = d->exp2alpha * d->expd3;
}

template <typename Scalar>
std::shared_ptr<typename LogCholeskyParametrizationTpl<
    Scalar>::InertialParametrizationDataAbstract>
LogCholeskyParametrizationTpl<Scalar>::createData() {
  return std::allocate_shared<LogCholeskyParametrizationData>(
      Eigen::aligned_allocator<LogCholeskyParametrizationData>());
}

template <typename Scalar>
bool LogCholeskyParametrizationTpl<Scalar>::checkData(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data) const {
  return std::dynamic_pointer_cast<LogCholeskyParametrizationData>(data) !=
         nullptr;
}

template <typename Scalar>
template <typename NewScalar>
LogCholeskyParametrizationTpl<NewScalar>
LogCholeskyParametrizationTpl<Scalar>::cast() const {
  return LogCholeskyParametrizationTpl<NewScalar>();
}

template <typename Scalar>
void LogCholeskyParametrizationTpl<Scalar>::print(std::ostream& os) const {
  os << "LogCholeskyParametrization";
}

}  // namespace crocoddyl
