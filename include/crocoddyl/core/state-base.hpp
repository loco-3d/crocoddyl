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
#include <vector>

namespace crocoddyl {

enum Jcomponent { both = 0, first = 1, second = 2 };

inline bool is_a_Jcomponent(Jcomponent firstsecond) {
  return (firstsecond == first || firstsecond == second || firstsecond == both);
}

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
                     Jcomponent firstsecond = both) = 0;
  virtual void Jintegrate(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& dx,
                          Eigen::Ref<Eigen::MatrixXd> Jfirst, Eigen::Ref<Eigen::MatrixXd> Jsecond,
                          Jcomponent firstsecond = both) = 0;

  const unsigned int& get_nx() const;
  const unsigned int& get_ndx() const;
  const unsigned int& get_nq() const;
  const unsigned int& get_nv() const;

 protected:
  unsigned int nx_;
  unsigned int ndx_;
  unsigned int nq_;
  unsigned int nv_;

#ifdef PYTHON_BINDINGS
 public:
  Eigen::VectorXd diff_wrap(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1) {
    Eigen::VectorXd dxout = Eigen::VectorXd::Zero(ndx_);
    diff(x0, x1, dxout);
    return dxout;
  }
  Eigen::VectorXd integrate_wrap(const Eigen::VectorXd& x, const Eigen::VectorXd& dx) {
    Eigen::VectorXd xout = Eigen::VectorXd::Zero(nx_);
    integrate(x, dx, xout);
    return xout;
  }
  std::vector<Eigen::MatrixXd> Jdiff_wrap(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1,
                                          std::string firstsecond = "both") {
    Eigen::MatrixXd Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
    std::vector<Eigen::MatrixXd> Jacs;
    if (firstsecond == "both") {
      Jdiff(x0, x1, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    } else if (firstsecond == "first") {
      Jdiff(x0, x1, Jfirst, Jsecond, first);
      Jacs.push_back(Jfirst);
    } else if (firstsecond == "second") {
      Jdiff(x0, x1, Jfirst, Jsecond, second);
      Jacs.push_back(Jsecond);
    } else {
      Jdiff(x0, x1, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    }
    return Jacs;
  }
  std::vector<Eigen::MatrixXd> Jintegrate_wrap(const Eigen::VectorXd& x, const Eigen::VectorXd& dx,
                                               std::string firstsecond = "both") {
    Eigen::MatrixXd Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
    std::vector<Eigen::MatrixXd> Jacs;
    if (firstsecond == "both") {
      Jintegrate(x, dx, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    } else if (firstsecond == "first") {
      Jintegrate(x, dx, Jfirst, Jsecond, first);
      Jacs.push_back(Jfirst);
    } else if (firstsecond == "second") {
      Jintegrate(x, dx, Jfirst, Jsecond, second);
      Jacs.push_back(Jsecond);
    } else {
      Jintegrate(x, dx, Jfirst, Jsecond, both);
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
    }
    return Jacs;
  }
#endif
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_STATE_BASE_HPP_
