///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ExpEigenValueParametrizationDataTpl<
    Scalar>::ExpEigenValueParametrizationDataTpl()
    : R(Matrix3s::Identity()),
      RD(Matrix3s::Zero()),
      I(Matrix3s::Zero()),
      Sh(Matrix3s::Zero()),
      D(Vector3s::Zero()),
      Lx(0.),
      Ly(0.),
      Lz(0.) {}

template <typename Scalar>
std::shared_ptr<typename ExpEigenValueParametrizationTpl<
    Scalar>::ExpEigenValueParametrizationData>
ExpEigenValueParametrizationTpl<Scalar>::castData(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data) const {
  std::shared_ptr<ExpEigenValueParametrizationData> d =
      std::dynamic_pointer_cast<ExpEigenValueParametrizationData>(data);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: data is not an ExpEigenValueParametrizationData");
  }
  return d;
}

template <typename Scalar>
void ExpEigenValueParametrizationTpl<Scalar>::fromParametrization(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data,
    Eigen::Ref<VectorXs> psi, const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != Base::kDimension ||
      static_cast<std::size_t>(psi.size()) != Base::kDimension) {
    throw_pretty("Invalid argument: p and psi must have dimension 10");
  }
  const std::shared_ptr<ExpEigenValueParametrizationData> d = castData(data);

  using std::exp;
  psi[0] = exp(p[0]);
  psi.segment(1, 3) = p.segment(1, 3);
  d->R = pinocchio::exp3(p.segment(4, 3));
  d->Lx = exp(p[7]);
  d->Ly = exp(p[8]);
  d->Lz = exp(p[9]);
  d->D << d->Ly + d->Lz, d->Lx + d->Lz, d->Lx + d->Ly;
  d->RD = d->R * d->D.asDiagonal();
  d->I = d->RD * d->R.transpose();
  const Vector3s h = psi.segment(1, 3);
  d->Sh = pinocchio::skew(h);
  d->I.noalias() += d->Sh.transpose() * d->Sh / psi[0];

  psi[4] = d->I(0, 0);
  psi[5] = d->I(0, 1);
  psi[6] = d->I(1, 1);
  psi[7] = d->I(0, 2);
  psi[8] = d->I(1, 2);
  psi[9] = d->I(2, 2);
}

template <typename Scalar>
void ExpEigenValueParametrizationTpl<Scalar>::toParametrization(
    Eigen::Ref<VectorXs> p, const Eigen::Ref<const VectorXs>& psi) {
  if (static_cast<std::size_t>(p.size()) != Base::kDimension ||
      static_cast<std::size_t>(psi.size()) != Base::kDimension) {
    throw_pretty("Invalid argument: p and psi must have dimension 10");
  }

  Matrix3s I;
  I << psi[4], psi[5], psi[7], psi[5], psi[6], psi[8], psi[7], psi[8], psi[9];
  const Vector3s h = psi.segment(1, 3);
  const Matrix3s Sh = pinocchio::skew(h);
  I.noalias() -= Sh.transpose() * Sh / psi[0];

  Eigen::JacobiSVD<Matrix3s> svd(I, Eigen::ComputeFullU);
  Matrix3s R = svd.matrixU();
  if (R.determinant() < Scalar(0.)) {
    R.col(2) *= Scalar(-1.);
  }
  const Vector3s D = svd.singularValues();

  using std::log;
  p[0] = log(psi[0]);
  p.segment(1, 3) = psi.segment(1, 3);
  p.segment(4, 3) = pinocchio::log3(R);
  p[7] = log((D[1] + D[2] - D[0]) / Scalar(2.));
  p[8] = log((D[0] + D[2] - D[1]) / Scalar(2.));
  p[9] = log((D[0] + D[1] - D[2]) / Scalar(2.));
}

template <typename Scalar>
void ExpEigenValueParametrizationTpl<Scalar>::updateParametrizationDerivative(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data,
    Eigen::Ref<MatrixXs> dpsi_dp, const Eigen::Ref<const VectorXs>& p,
    const Eigen::Ref<const VectorXs>& psi) {
  if (static_cast<std::size_t>(p.size()) != Base::kDimension ||
      static_cast<std::size_t>(psi.size()) != Base::kDimension ||
      static_cast<std::size_t>(dpsi_dp.rows()) != Base::kDimension ||
      static_cast<std::size_t>(dpsi_dp.cols()) != Base::kDimension) {
    throw_pretty("Invalid argument: p, psi and dpsi_dp must have dimension 10");
  }
  const std::shared_ptr<ExpEigenValueParametrizationData> d = castData(data);
  dpsi_dp.setZero();

  const Scalar m = psi[0];
  const Vector3s h = psi.segment(1, 3);
  const Vector3s w = p.segment(4, 3);
  using std::exp;
  const Scalar Lx = exp(p[7]);
  const Scalar Ly = exp(p[8]);
  const Scalar Lz = exp(p[9]);

  d->R = pinocchio::exp3(w);
  d->D << Ly + Lz, Lx + Lz, Lx + Ly;
  const Matrix3s Lmat = d->D.asDiagonal();

  const Scalar r2 = h.dot(h);
  const Matrix3s P = (r2 * Matrix3s::Identity() - h * h.transpose()) / m;
  Matrix3s Jw;
  pinocchio::Jexp3(w, Jw);

  Matrix3s dIc_dw[3];
  for (std::size_t k = 0; k < 3; ++k) {
    const Vector3s v = Jw.col(static_cast<Eigen::Index>(k));
    const Matrix3s S = pinocchio::skew(v);
    const Matrix3s H = S * Lmat + Lmat * S.transpose();
    dIc_dw[k] = d->R * H * d->R.transpose();
  }

  const Matrix3s dIc_dLx =
      d->R * (Vector3s(Scalar(0.), Scalar(1.), Scalar(1.)).asDiagonal()) *
      d->R.transpose();
  const Matrix3s dIc_dLy =
      d->R * (Vector3s(Scalar(1.), Scalar(0.), Scalar(1.)).asDiagonal()) *
      d->R.transpose();
  const Matrix3s dIc_dLz =
      d->R * (Vector3s(Scalar(1.), Scalar(1.), Scalar(0.)).asDiagonal()) *
      d->R.transpose();

  dpsi_dp(0, 0) = m;
  dpsi_dp(1, 1) = Scalar(1.);
  dpsi_dp(2, 2) = Scalar(1.);
  dpsi_dp(3, 3) = Scalar(1.);

  for (std::size_t j = 0; j < Base::kDimension; ++j) {
    Matrix3s dI = Matrix3s::Zero();

    if (j == 0) {
      dI -= P;
    } else if (j >= 1 && j <= 3) {
      Vector3s e = Vector3s::Zero();
      e[static_cast<Eigen::Index>(j - 1)] = Scalar(1.);
      const Matrix3s dA = Scalar(2.) * h[static_cast<Eigen::Index>(j - 1)] *
                              Matrix3s::Identity() -
                          e * h.transpose() - h * e.transpose();
      dI = dA / m;
    } else if (j >= 4 && j <= 6) {
      dI = dIc_dw[j - 4];
    } else if (j == 7) {
      dI = Lx * dIc_dLx;
    } else if (j == 8) {
      dI = Ly * dIc_dLy;
    } else if (j == 9) {
      dI = Lz * dIc_dLz;
    }

    dpsi_dp(4, static_cast<Eigen::Index>(j)) = dI(0, 0);
    dpsi_dp(5, static_cast<Eigen::Index>(j)) = dI(0, 1);
    dpsi_dp(6, static_cast<Eigen::Index>(j)) = dI(1, 1);
    dpsi_dp(7, static_cast<Eigen::Index>(j)) = dI(0, 2);
    dpsi_dp(8, static_cast<Eigen::Index>(j)) = dI(1, 2);
    dpsi_dp(9, static_cast<Eigen::Index>(j)) = dI(2, 2);
  }
}

template <typename Scalar>
std::shared_ptr<typename ExpEigenValueParametrizationTpl<
    Scalar>::InertialParametrizationDataAbstract>
ExpEigenValueParametrizationTpl<Scalar>::createData() {
  return std::allocate_shared<ExpEigenValueParametrizationData>(
      Eigen::aligned_allocator<ExpEigenValueParametrizationData>());
}

template <typename Scalar>
bool ExpEigenValueParametrizationTpl<Scalar>::checkData(
    const std::shared_ptr<InertialParametrizationDataAbstract>& data) const {
  return std::dynamic_pointer_cast<ExpEigenValueParametrizationData>(data) !=
         nullptr;
}

template <typename Scalar>
template <typename NewScalar>
ExpEigenValueParametrizationTpl<NewScalar>
ExpEigenValueParametrizationTpl<Scalar>::cast() const {
  return ExpEigenValueParametrizationTpl<NewScalar>();
}

template <typename Scalar>
void ExpEigenValueParametrizationTpl<Scalar>::print(std::ostream& os) const {
  os << "ExpEigenValueParametrization";
}

}  // namespace crocoddyl
