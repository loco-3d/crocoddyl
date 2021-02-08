///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief This class encapsulates a friction cone
 *
 * A friction cone is a 3D cone that characterizes feasible contact wrench.
 * The friction cone defines a linearized version (pyramid) with a predefined number of facets.
 *
 * /sa `WrenchConeTpl`
 */
template <typename _Scalar>
class FrictionConeTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::Quaternions Quaternions;

  /**
   * @brief Initialize the friction cone
   */
  explicit FrictionConeTpl();

  /**
   * @brief Initialize the wrench cone
   *
   * @param[in] R           Rotation matrix that defines the cone orientation w.r.t. the inertial frame
   * @param[in] mu          Friction coefficient
   * @param[in] nf          Number of facets (default 4)
   * @param[in] inner_appr  Label that describes the type of friction cone approximation (inner/outer)
   * @param[in] min_nforce  Minimum normal force (default 0.)
   * @param[in] max_nforce  Maximum normal force (default maximum floating number))
   */
  FrictionConeTpl(const Matrix3s& R, const Scalar mu, std::size_t nf = 4, const bool inner_appr = true,
                  const Scalar min_nforce = Scalar(0.), const Scalar max_nforce = std::numeric_limits<Scalar>::max());
  DEPRECATED("Use constructor based on rotation matrix.",
             FrictionConeTpl(const Vector3s& normal, const Scalar mu, std::size_t nf = 4, const bool inner_appr = true,
                             const Scalar min_nforce = Scalar(0.),
                             const Scalar max_nforce = std::numeric_limits<Scalar>::max());)

  /**
   * @brief Initialize the wrench cone
   *
   * @param[in] cone  Friction cone
   */
  FrictionConeTpl(const FrictionConeTpl<Scalar>& cone);
  ~FrictionConeTpl();

  /**
   * @brief Update the matrix and bound of friction cone inequalities in the world frame.
   *
   * This matrix-vector pair describes the linearized Coulomb friction model as follow:
   * \f$ -ub \leq A \times w \leq -lb \f$,
   * where wrench, \f$ w \f$, is expressed in the inertial frame located with axes parallel to
   * those of the world frame.
   */
  void update();
  DEPRECATED("Use update()", void update(const Vector3s& normal, const Scalar mu, const bool inner_appr = true,
                                         const Scalar min_nforce = Scalar(0.),
                                         const Scalar max_nforce = std::numeric_limits<Scalar>::max()));

  /**
   * @brief Return the matrix of friction cone
   */
  const MatrixX3s& get_A() const;

  /**
   * @brief Return the upper bound of the friction cone
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Return the lower bound of the friction cone
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the number of facets
   */
  std::size_t get_nf() const;

  /**
   * @brief Return the surface normal vector
   */
  const Matrix3s& get_R() const;

  DEPRECATED("Use get_R.", Vector3s get_nsurf();)

  /**
   * @brief Return the friction coefficient
   */
  const Scalar get_mu() const;

  /**
   * @brief Return the label that describes the type of friction cone approximation (inner/outer)
   */
  bool get_inner_appr() const;

  /**
   * @brief Return the minimum normal force
   */
  const Scalar get_min_nforce() const;

  /**
   * @brief Return the maximum normal force
   */
  const Scalar get_max_nforce() const;

  /**
   * @brief Modify the rotation matrix that defines the cone orientation w.r.t. the inertial frame
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_R(const Matrix3s& R);

  DEPRECATED("Use set_R.", void set_nsurf(const Vector3s& nsurf);)

  /**
   * @brief Modify friction coefficient
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_mu(const Scalar mu);

  /**
   * @brief Modify the label that describes the type of friction cone approximation (inner/outer)
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_inner_appr(const bool inner_appr);

  /**
   * @brief Modify the maximum normal force
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_min_nforce(const Scalar min_nforce);

  /**
   * @brief Modify the maximum normal force
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_max_nforce(const Scalar max_nforce);

  FrictionConeTpl<Scalar>& operator=(const FrictionConeTpl<Scalar>& other);

  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const FrictionConeTpl<Scalar>& X);

 private:
  std::size_t nf_;     //!< Number of facets
  MatrixX3s A_;        //!< Matrix of friction cone
  VectorXs ub_;        //!< Upper bound of the friction cone
  VectorXs lb_;        //!< Lower bound of the friction cone
  Matrix3s R_;         //!< Rotation of the friction cone w.r.t. the inertial frame
  Scalar mu_;          //!< Friction coefficient
  bool inner_appr_;    //!< Label that describes the type of friction cone approximation (inner/outer)
  Scalar min_nforce_;  //!< Minimum normal force
  Scalar max_nforce_;  //!< Maximum normal force
};

}  // namespace crocoddyl
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/friction-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_FRICTION_CONE_HPP_
