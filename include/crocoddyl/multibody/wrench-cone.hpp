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

/**
 * @brief This class encapsulates a wrench cone
 *
 * A wrench cone is a 6D polyhedral convex cone that chracterizes feasible contact wrench.
 * The wrench cone is derived in the case of rectangular support areas, which is of practical importance since most
 * humanoid robot feet can be adequately approximated by rectangles.
 */

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

  /**
   * @brief Initialize the wrench cone
   */
  explicit WrenchConeTpl();

  /**
   * @brief Initialize the wrench cone
   *
   * @param[in] R               Rotation matrix that defines the cone orientation
   * @param[in] mu              Friction coefficient
   * @param[in] box             Dimension of the foot surface dim = (length, width)
   * @param[in] nf              Number of facets (default 16)
   * @param[in] min_nforce      Minimum normal force (default 0.)
   * @param[in] max_nforce      Maximum normal force (default default sys.float_info.max))
   */
  WrenchConeTpl(const Matrix3s& R, const Scalar& mu, const Vector2s& box_size, std::size_t nf = 16,
                const Scalar& min_nforce = Scalar(0.), const Scalar& max_nforce = std::numeric_limits<Scalar>::max());

  /**
   * @brief Initialize the wrench cone
   *
   * @param[in] cone            Wrench cone
   */
  WrenchConeTpl(const WrenchConeTpl<Scalar>& cone);
  ~WrenchConeTpl();

  /**
   * @brief Update the matrix of wrench cone inequaliteis in the world frame.
   *
   * This matrix-vector pair describes the linearized Coulomb friction model as follow:
   *
   * \f$ -ub \leq A \times w \leq -lb \f$,
   *
   * where wrench, \f$ w \f$, is expressed in the inertial frame located at the
   * center of the rectangular foot contact area (length, width) with axes parallel to
   * those of the world frame.
   *
   * @param[in] R               Rotation matrix that defines the cone orientation
   * @param[in] mu              Friction coefficient
   * @param[in] box             Dimension of the foot surface dim = (length, width)
   * @param[in] min_nforce      Minimum normal force (default 0.)
   * @param[in] max_nforce      Maximum normal force (default default sys.float_info.max))
   */
  void update(const Matrix3s& R, const Scalar& mu, const Vector2s& box_size, const Scalar& min_nforce = Scalar(0.),
              const Scalar& max_nforce = std::numeric_limits<Scalar>::max());

  /**
   * @brief Return the matrix of wrench cone
   */
  const MatrixX6s& get_A() const;

  /**
   * @brief Return the lower bound of inequalities
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the upper bound of inequalities
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Return the rotation matrix that defines the cone orientation
   */
  const Matrix3s& get_R() const;

  /**
   * @brief Return dimension of the foot surface dim = (length, width)
   */
  const Vector2s& get_box() const;

  /**
   * @brief Return friction coefficient
   */
  const Scalar& get_mu() const;

  /**
   * @brief Return the number of facets
   */
  const std::size_t& get_nf() const;

  /**
   * @brief Return the minimum normal force
   */
  const Scalar& get_min_nforce() const;

  /**
   * @brief Return the maximum normal force
   */
  const Scalar& get_max_nforce() const;

  /**
   * @brief Modify the rotation matrix that defines the cone orientation
   */
  void set_R(Matrix3s R);

  /**
   * @brief Modify dimension of the foot surface dim = (length, width)
   */
  void set_box(Vector2s box);

  /**
   * @brief Modify friction coefficient
   */
  void set_mu(Scalar mu);

  /**
   * @brief Modify the minium normal force
   */
  void set_min_nforce(Scalar min_nforce);

  /**
   * @brief Modify the maximum normal force
   */
  void set_max_nforce(Scalar max_nforce);

  /**
   * @brief Modify the maximum normal force
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const WrenchConeTpl<Scalar>& X);

 private:
  MatrixX6s A_;
  VectorXs ub_;
  VectorXs lb_;
  Matrix3s R_;
  Vector2s box_;
  Scalar mu_;
  std::size_t nf_;
  Scalar min_nforce_;
  Scalar max_nforce_;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/wrench-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_WRENCH_CONE_HPP_
