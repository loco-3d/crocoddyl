///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_STATE_HPP_
#define CROCODDYL_CORE_NUMDIFF_STATE_HPP_

#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

class StateNumDiff : public StateAbstract {
 public:
  explicit StateNumDiff(StateAbstract& state);
  ~StateNumDiff();

  Eigen::VectorXd zero();
  Eigen::VectorXd rand();
  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout);
  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> xout);
  /**
   * @brief This computes the Jacobian of the diff method by finite
   * differentiation:
   * \f{equation}{
   *    Jfirst[:,k] = diff(int(x_1, dx_dist), x_2) - diff(x_1, x_2)/disturbance
   * \f}
   * and
   * \f{equation}{
   *    Jsecond[:,k] = diff(x_1, int(x_2, dx_dist)) - diff(x_1, x_2)/disturbance
   * \f}
   *
   * @param Jfirst
   * @param Jsecond
   * @param firstsecond
   */
  void Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond, Jcomponent firstsecond = both);
  /**
   * @brief This computes the Jacobian of the integrate method by finite
   * differentiation:
   * \f{equation}{
   *    Jfirst[:,k] = diff( int(x, d_x), int( int(x, dx_dist), dx) )/disturbance
   * \f}
   * and
   * \f{equation}{
   *    Jsecond[:,k] = diff( int(x, d_x), int( x, dx + dx_dist) )/disturbance
   * \f}
   *
   * @param Jfirst
   * @param Jsecond
   * @param firstsecond
   */
  void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                  Jcomponent firstsecond = both);
  const double& get_disturbance() { return disturbance_; }

 private:
  /**
   * @brief This is the state we need to compute the numerical differentiation
   * from.
   */
  StateAbstract& state_;
  /**
   * @brief This the increment used in the finite differentiation and integration.
   */
  double disturbance_;
  /**
   * @brief This is the vector containing the small element during the finite
   * differentiation and integration. This is a temporary variable but used
   * quiet often. For sake of memory management we allocate it once in the
   * constructor of this class.
   */
  Eigen::VectorXd dx_;
  /**
   * @brief This is the result of diff(x0, x1, dx0_) in the finite
   * differentiation. This is the state difference around which to compute
   * the jacobians of the finite difference.
   */
  Eigen::VectorXd dx0_;
  /**
   * @brief This is the result of integrate(x, dx x0_) in the finite
   * integration. This is the state around which to compute the jacobians of
   * the finite integrate
   */
  Eigen::VectorXd x0_;
  /**
   * @brief This is the vector containing the result of an integration. This is
   * a temporary variable but used quiet often. For sake of memory management we
   * allocate it once in the constructor of this class.
   */
  Eigen::VectorXd tmp_x_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_STATE_HPP_
