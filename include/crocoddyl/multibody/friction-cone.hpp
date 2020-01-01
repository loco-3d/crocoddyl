///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_

#include <Eigen/Dense>

namespace crocoddyl {

class FrictionCone {
 public:
  typedef Eigen::Matrix<double, Eigen::Dynamic, 3> MatrixX3;

  FrictionCone(const Eigen::Vector3d& normal, const double& mu, std::size_t nf = 4, bool inner_appr = true,
               const double& min_nforce = 0., const double& max_nforce = std::numeric_limits<double>::max());
  FrictionCone(const FrictionCone& cone);
  ~FrictionCone();

  void update(const Eigen::Vector3d& normal, const double& mu, bool inner_appr = true, const double& min_nforce = 0.,
              const double& max_nforce = std::numeric_limits<double>::max());

  const MatrixX3& get_A() const;
  const Eigen::VectorXd& get_lb() const;
  const Eigen::VectorXd& get_ub() const;
  const Eigen::Vector3d& get_nsurf() const;
  const double& get_mu() const;
  const std::size_t& get_nf() const;
  const bool& get_inner_appr() const;
  const double& get_min_nforce() const;
  const double& get_max_nforce() const;

 private:
  MatrixX3 A_;
  Eigen::VectorXd lb_;
  Eigen::VectorXd ub_;
  Eigen::Vector3d nsurf_;
  double mu_;
  std::size_t nf_;
  bool inner_appr_;
  double min_nforce_;
  double max_nforce_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
