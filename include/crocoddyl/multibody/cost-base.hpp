///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace crocoddyl {

struct CostDataAbstract;  // forward declaration

class CostModelAbstract {
 public:
  CostModelAbstract(pinocchio::Model* const model, ActivationModelAbstract* const activation, const unsigned int& nu,
                    const bool& with_residuals = true);
  CostModelAbstract(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                    const bool& with_residuals = true);
  CostModelAbstract(pinocchio::Model* const model, const unsigned int& nr, const unsigned int& nu,
                    const bool& with_residuals = true);
  CostModelAbstract(pinocchio::Model* const model, const unsigned int& nr, const bool& with_residuals = true);
  ~CostModelAbstract();

  virtual void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) = 0;
  virtual boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  pinocchio::Model* get_pinocchio() const;
  ActivationModelAbstract* get_activation() const;
  const unsigned int& get_nq() const;
  const unsigned int& get_nv() const;
  const unsigned int& get_nu() const;
  const unsigned int& get_nx() const;
  const unsigned int& get_ndx() const;
  const unsigned int& get_nr() const;

 protected:
  pinocchio::Model* pinocchio_;
  ActivationModelAbstract* activation_;
  unsigned int nq_;
  unsigned int nv_;
  unsigned int nu_;
  unsigned int nx_;
  unsigned int ndx_;
  unsigned int nr_;
  bool with_residuals_;
  Eigen::VectorXd unone_;

#ifdef PYTHON_BINDINGS
 public:
  void calc_wrap(boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    calc(data, x, u);
  }
  void calc_wrap(boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x) { calc(data, x, unone_); }

  void calcDiff_wrap(boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                     const bool& recalc) {
    calcDiff(data, x, u, recalc);
  }
  void calcDiff_wrap(boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    calcDiff(data, x, u, true);
  }
  void calcDiff_wrap(boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_, true);
  }
  void calcDiff_wrap(boost::shared_ptr<CostDataAbstract>& data, const Eigen::VectorXd& x, const bool& recalc) {
    calcDiff(data, x, unone_, recalc);
  }
#endif
};

struct CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataAbstract(Model* const model, pinocchio::Data* const data)
      : pinocchio(data), activation(model->get_activation()->createData()), cost(0.) {
    const int& ndx = model->get_ndx();
    const int& nu = model->get_nu();
    const int& nr = model->get_nr();
    Lx = Eigen::VectorXd::Zero(ndx);
    Lu = Eigen::VectorXd::Zero(nu);
    Lxx = Eigen::MatrixXd::Zero(ndx, ndx);
    Lxu = Eigen::MatrixXd::Zero(ndx, nu);
    Luu = Eigen::MatrixXd::Zero(nu, nu);
    r = Eigen::VectorXd::Zero(nr);
    Rx = Eigen::MatrixXd::Zero(nr, ndx);
    Ru = Eigen::MatrixXd::Zero(nr, nu);
  }

  pinocchio::Data* get_pinocchio() const { return pinocchio; }
  boost::shared_ptr<ActivationDataAbstract> get_activation() const { return activation; }
  const double& get_cost() const { return cost; }
  const Eigen::VectorXd& get_Lx() const { return Lx; }
  const Eigen::VectorXd& get_Lu() const { return Lu; }
  const Eigen::MatrixXd& get_Lxx() const { return Lxx; }
  const Eigen::MatrixXd& get_Lxu() const { return Lxu; }
  const Eigen::MatrixXd& get_Luu() const { return Luu; }
  const Eigen::VectorXd& get_r() const { return r; }
  const Eigen::MatrixXd& get_Rx() const { return Rx; }
  const Eigen::MatrixXd& get_Ru() const { return Ru; }

  pinocchio::Data* pinocchio;
  boost::shared_ptr<ActivationDataAbstract> activation;
  double cost;
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

#endif  // CROCODDYL_MULTIBODY_COST_BASE_HPP_