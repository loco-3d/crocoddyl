///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <iostream>
#include "crocoddyl/multibody/friction-cone.hpp"

namespace crocoddyl {

FrictionCone::FrictionCone(const Eigen::Vector3d& normal, const double& mu, std::size_t nf, bool inner_appr,
                           const double& min_nforce, const double& max_nforce)
    : nf_(nf) {
  if (nf_ % 2 != 0) {
    nf_ = 4;
    std::cerr << "Warning: nf has to be an even number, set to 4" << std::endl;
  }
  A_.resize(nf_ + 1, 3);
  lb_.resize(nf_ + 1);
  ub_.resize(nf_ + 1);

  // compute the matrix
  update(normal, mu, inner_appr, min_nforce, max_nforce);
}

FrictionCone::~FrictionCone() {}

void FrictionCone::update(const Eigen::Vector3d& normal, const double& mu, bool inner_appr, const double& min_nforce,
                          const double& max_nforce) {
  nsurf_ = normal;
  mu_ = mu;
  inner_appr_ = inner_appr;
  min_nforce_ = min_nforce;
  max_nforce_ = max_nforce;

  // Sanity checks
  if (normal.norm() != 1.) {
    nsurf_ /= normal.norm();
    std::cerr << "Warning: normal is not an unitary vector, then we normalized it" << std::endl;
  }
  if (min_nforce < 0.) {
    min_nforce_ = 0.;
    std::cerr << "Warning: min_nforce has to be a positive value, set to 0" << std::endl;
  }
  if (max_nforce < 0.) {
    max_nforce_ = std::numeric_limits<double>::max();
    std::cerr << "Warning: max_nforce has to be a positive value, set to maximun value" << std::endl;
  }

  double theta = 2 * M_PI / static_cast<double>(nf_);
  if (inner_appr_) {
    mu_ *= cos(theta / 2.);
  }

  Eigen::Matrix3d c_R_o = Eigen::Quaterniond::FromTwoVectors(nsurf_, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  for (std::size_t i = 0; i < nf_ / 2; ++i) {
    double theta_i = theta * static_cast<double>(i);
    Eigen::Vector3d tsurf_i = Eigen::Vector3d(cos(theta_i), sin(theta_i), 0.);
    A_.row(2 * i) = (-mu_ * Eigen::Vector3d::UnitZ() + tsurf_i).transpose() * c_R_o;
    A_.row(2 * i + 1) = (-mu_ * Eigen::Vector3d::UnitZ() - tsurf_i).transpose() * c_R_o;
    lb_(2 * i) = -std::numeric_limits<double>::max();
    lb_(2 * i + 1) = -std::numeric_limits<double>::max();
    ub_(2 * i) = 0.;
    ub_(2 * i + 1) = 0.;
  }
  A_.row(nf_) = nsurf_.transpose();
  lb_(nf_) = min_nforce_;
  ub_(nf_) = max_nforce_;
}

const FrictionCone::MatrixX3& FrictionCone::get_A() const { return A_; }

const Eigen::VectorXd& FrictionCone::get_lb() const { return lb_; }

const Eigen::VectorXd& FrictionCone::get_ub() const { return ub_; }

const Eigen::Vector3d& FrictionCone::get_nsurf() const { return nsurf_; }

const double& FrictionCone::get_mu() const { return mu_; }

const std::size_t& FrictionCone::get_nf() const { return nf_; }

const bool& FrictionCone::get_inner_appr() const { return inner_appr_; }

const double& FrictionCone::get_min_nforce() const { return min_nforce_; }

const double& FrictionCone::get_max_nforce() const { return max_nforce_; }

}  // namespace crocoddyl
