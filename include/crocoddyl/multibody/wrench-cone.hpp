///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_WRENCH_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

template <typename _Scalar>
class WrenchConeTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::MatrixX6s MatrixX6s;
  typedef typename MathBase::Quaternions Quaternions;

  explicit WrenchConeTpl();
  WrenchConeTpl(const Matrix3s& rot, const Scalar& mu, const Vector2s& box_size, std::size_t nf = 16);
  WrenchConeTpl(const WrenchConeTpl<Scalar>& cone);
  ~WrenchConeTpl();

  void update(const Matrix3s& rot, const Scalar& mu, const Vector2s& box_size);

  const MatrixX6s& get_A() const;
  const VectorXs& get_lb() const;
  const VectorXs& get_ub() const;
  const Matrix3s& get_rot() const;
  const Vector2s& get_box() const;
  const Scalar& get_mu() const;
  const std::size_t& get_nf() const;

  void set_rot(Matrix3s rot);
  void set_box(Vector2s box);
  void set_mu(Scalar mu);

  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const WrenchConeTpl<Scalar>& X);

 private:
  MatrixX6s A_;
  VectorXs ub_;
  VectorXs lb_;
  Matrix3s rot_;
  Vector2s box_;
  Scalar mu_;
  std::size_t nf_;
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/wrench-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_WRENCH_CONE_HPP_
