///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_STATE_BASE_HPP_
#define CROCODDYL_CORE_STATE_BASE_HPP_

#include <Eigen/Core>

namespace crocoddyl {

enum Jcomponent { both = 0, first, second };

class StateAbstract {
 public:
  StateAbstract(const unsigned int& nx, const unsigned int& ndx);
  virtual ~StateAbstract();

  virtual Eigen::VectorXd zero() = 0;
  virtual Eigen::VectorXd rand() = 0;
  virtual void diff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                    Eigen::Ref<Eigen::VectorXd> dxout) = 0;
  virtual void integrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                         Eigen::Ref<Eigen::VectorXd> xout) = 0;
  virtual void Jdiff(const Eigen::Ref<const Eigen::VectorXd>& x0, const Eigen::Ref<const Eigen::VectorXd>& x1,
                     Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                     Jcomponent firstsecond = Jcomponent::both) = 0;
  virtual void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                          Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                          Jcomponent firstsecond = Jcomponent::both) = 0;

  const unsigned int& get_nx() const;
  const unsigned int& get_ndx() const;

 protected:
  unsigned int nx_;
  unsigned int ndx_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_STATE_BASE_HPP_
