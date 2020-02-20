///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

template <typename _Scalar>
class FrictionConeTpl {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::Quaternions Quaternions;

  FrictionConeTpl(const Vector3s& normal, const Scalar& mu, std::size_t nf = 4, bool inner_appr = true,
                  const Scalar& min_nforce = 0., const Scalar& max_nforce = std::numeric_limits<Scalar>::max());
  FrictionConeTpl(const FrictionConeTpl<Scalar>& cone);
  ~FrictionConeTpl();

  void update(const Vector3s& normal, const Scalar& mu, bool inner_appr = true, const Scalar& min_nforce = 0.,
              const Scalar& max_nforce = std::numeric_limits<Scalar>::max());

  const MatrixX3s& get_A() const;
  const VectorXs& get_lb() const;
  const VectorXs& get_ub() const;
  const Vector3s& get_nsurf() const;
  const Scalar& get_mu() const;
  const std::size_t& get_nf() const;
  const bool& get_inner_appr() const;
  const Scalar& get_min_nforce() const;
  const Scalar& get_max_nforce() const;

 private:
  MatrixX3s A_;
  VectorXs lb_;
  VectorXs ub_;
  Vector3s nsurf_;
  Scalar mu_;
  std::size_t nf_;
  bool inner_appr_;
  Scalar min_nforce_;
  Scalar max_nforce_;
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/friction-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
