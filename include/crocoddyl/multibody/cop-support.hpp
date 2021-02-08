///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COP_SUPPORT_HPP_
#define CROCODDYL_MULTIBODY_COP_SUPPORT_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

/**
 * @brief This class encapsulates a center of pressure support of a 6d contact
 *
 * A CoP support defines a rectangular region that characterizes feasible CoP position.
 *
 * /sa `FrictionConeTpl`, `WrenchConeTpl`
 */
template <typename _Scalar>
class CoPSupportTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Vector4s Vector4s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix46s Matrix46s;
  typedef typename MathBase::Quaternions Quaternions;

  /**
   * @brief Initialize the center of pressure support
   */
  explicit CoPSupportTpl();

  /**
   * @brief Initialize the center of pressure support
   *
   * @param[in] R    Rotation matrix that defines the support orientation w.r.t. the inertial frame
   * @param[in] box  Dimension of the foot surface dim = (length, width)
   */
  CoPSupportTpl(const Matrix3s& R, const Vector2s& box);

  /**
   * @brief Initialize the center of pressure support
   *
   * @param[in] support  Center of pressure support
   */
  CoPSupportTpl(const WrenchConeTpl<Scalar>& support);
  ~CoPSupportTpl();

  /**
   * @brief Update the matrix of center of pressure inequalities in the world frame.
   *
   * This matrix-vector pair describes the center of pressure model as follow:
   * \f$ -ub \leq A \times w \leq -lb \f$,
   * where wrench, \f$ w \f$, is expressed in the inertial frame located at the
   * center of the rectangular foot contact area (length, width) with axes parallel to
   * those of the world frame.
   */
  void update();

  /**
   * @brief Return the matrix of center of pressure support
   */
  const Matrix46s& get_A() const;

  /**
   * @brief Return the upper bound of the center of pressure support
   */
  const Vector4s& get_ub() const;

  /**
   * @brief Return the lower bound of the center of pressure support
   */
  const Vector4s& get_lb() const;

  /**
   * @brief Return dimension of the center of pressure support (length, width)
   */
  const Vector2s& get_box() const;

  /**
   * @brief Return the rotation matrix that defines the support orientation w.r.t. the inertial frame
   */
  const Matrix3s& get_R() const;

  /**
   * @brief Modify the rotation matrix that defines the support orientation w.r.t. the inertial frame
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_R(const Matrix3s& R);

  /**
   * @brief Modify dimension of the center of pressure support (length, width)
   *
   * Note that you need to run `update` for updating the inequality matrix and bounds.
   */
  void set_box(const Vector2s& box);

  CoPSupportTpl<Scalar>& operator=(const CoPSupportTpl<Scalar>& other);

  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os, const CoPSupportTpl<Scalar>& X);

 private:
  Matrix46s A_;   //!< Matrix of wrench cone
  Vector4s ub_;   //!< Upper bound of the wrench cone
  Vector4s lb_;   //!< Lower bound of the wrench cone
  Matrix3s R_;    //!< Rotation of the wrench cone w.r.t. the inertial frame
  Vector2s box_;  //!< Dimension of the foot surface (length, width)
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/cop-support.hxx"

#endif  // CROCODDYL_MULTIBODY_COP_SUPPORT_HPP_
