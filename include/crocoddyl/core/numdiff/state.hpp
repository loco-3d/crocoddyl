///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_STATE_HPP_
#define CROCODDYL_CORE_NUMDIFF_STATE_HPP_

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include "crocoddyl/core/state-base.hpp"

namespace crocoddyl {

class StateNumDiff : public StateAbstract {
 public:
  explicit StateNumDiff(boost::shared_ptr<StateAbstract> state);
  ~StateNumDiff();

  Eigen::VectorXd zero() const;
  Eigen::VectorXd rand() const;
  void diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
            Eigen::Ref<Eigen::VectorXd> dxout) const;
  void integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                 Eigen::Ref<Eigen::VectorXd> xout) const;
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
             Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
             Jcomponent firstsecond = both) const;
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
                  Jcomponent firstsecond = both) const;
  const double& get_disturbance() const { return disturbance_; }

 private:
  /**
   * @brief This is the state we need to compute the numerical differentiation
   * from.
   */
  boost::shared_ptr<StateAbstract> state_;
  /**
   * @brief This the increment used in the finite differentiation and integration.
   */
  double disturbance_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_STATE_HPP_
