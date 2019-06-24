///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_ACTION_BASE_HPP_
#define CROCODDYL_CORE_ACTION_BASE_HPP_

#include <crocoddyl/core/state-base.hpp>
#include <memory>

//TODO: DifferentialActionModelAbstract DifferentialActionDataAbstract

namespace crocoddyl {

struct ActionDataAbstract; // forward declaration

class ActionModelAbstract {
 public:
  ActionModelAbstract(StateAbstract *const state,
                      const unsigned int& nu,
                      const unsigned int& ncost);
  ~ActionModelAbstract();

  virtual void calc(std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u,
                        const bool& recalc=true) = 0;
  virtual std::shared_ptr<ActionDataAbstract> createData() = 0;

  void calc(std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x);

  unsigned int get_nx() const;
  unsigned int get_ndx() const;
  unsigned int get_nu() const;
  unsigned int get_ncost() const;
  StateAbstract* get_state() const;

 protected:
  unsigned int nx_;
  unsigned int ndx_;
  unsigned int nu_;
  unsigned int ncost_;
  StateAbstract* state_;
  Eigen::VectorXd unone_;
};

struct ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template<typename Model>
  ActionDataAbstract(Model *const model) {
    const int& nx = model->get_nx();
    const int& ndx = model->get_ndx();
    const int& nu = model->get_nu();
    const int& ncost = model->get_ncost();
    xnext = Eigen::VectorXd::Zero(nx);
    Fx = Eigen::MatrixXd::Zero(ndx, ndx);
    Fu = Eigen::MatrixXd::Zero(ndx, nu);
    Lx = Eigen::VectorXd::Zero(ndx);
    Lu = Eigen::VectorXd::Zero(nu);
    Lxx = Eigen::MatrixXd::Zero(ndx, ndx);
    Lxu = Eigen::MatrixXd::Zero(ndx, nu);
    Luu = Eigen::MatrixXd::Zero(nu, nu);
    r = Eigen::VectorXd::Zero(ncost);
    Rx = Eigen::MatrixXd::Zero(ncost, ndx);
    Ru = Eigen::MatrixXd::Zero(ncost, nu);
  }

  const double& get_cost() { return cost; }
  const Eigen::VectorXd& get_xnext() { return xnext; }
  const Eigen::VectorXd& get_Lx() { return Lx; }
  const Eigen::VectorXd& get_Lu() { return Lu; }
  const Eigen::MatrixXd& get_Lxx() { return Lxx; }
  const Eigen::MatrixXd& get_Lxu() { return Lxu; }
  const Eigen::MatrixXd& get_Luu() { return Luu; }
  const Eigen::MatrixXd& get_Fx() { return Fx; }
  const Eigen::MatrixXd& get_Fu() { return Fu; }
  const Eigen::VectorXd& get_r() { return r; }
  const Eigen::MatrixXd& get_Rx() { return Rx; }
  const Eigen::MatrixXd& get_Ru() { return Ru; }

  double cost;
  Eigen::VectorXd xnext;
  Eigen::MatrixXd Fx;
  Eigen::MatrixXd Fu;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
  Eigen::VectorXd r;
  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTION_BASE_HPP_
