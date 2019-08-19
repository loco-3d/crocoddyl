///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_

#include <string>
#include <map>
#include <utility>
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

struct CostItem {
  CostItem() {}
  CostItem(const std::string& name, CostModelAbstract* cost, const double& weight)
      : name(name), cost(cost), weight(weight) {}

  std::string name;
  CostModelAbstract* cost;
  double weight;
};

struct CostDataSum;  // forward declaration

class CostModelSum {
 public:
  typedef std::map<std::string, CostItem> CostModelContainer;
  typedef std::map<std::string, boost::shared_ptr<CostDataAbstract> > CostDataContainer;

  CostModelSum(StateMultibody& state, unsigned int const& nu, const bool& with_residuals = true);
  explicit CostModelSum(StateMultibody& state, const bool& with_residuals = true);
  ~CostModelSum();

  void addCost(const std::string& name, CostModelAbstract* const cost, const double& weight);
  void removeCost(const std::string& name);

  void calc(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataSum> createData(pinocchio::Data* const data);

  void calc(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  StateMultibody& get_state() const;
  const CostModelContainer& get_costs() const;
  const unsigned int& get_nu() const;
  const unsigned int& get_nr() const;

 private:
  StateMultibody& state_;
  CostModelContainer costs_;
  unsigned int nu_;
  unsigned int nr_;
  bool with_residuals_;
  Eigen::VectorXd unone_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<CostDataSum>& data, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& u = Eigen::VectorXd()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
  }

  void calcDiff_wrap(const boost::shared_ptr<CostDataSum>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u,
                     const bool& recalc) {
    calcDiff(data, x, u, recalc);
  }
  void calcDiff_wrap(const boost::shared_ptr<CostDataSum>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    calcDiff(data, x, u, true);
  }
  void calcDiff_wrap(const boost::shared_ptr<CostDataSum>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_, true);
  }
  void calcDiff_wrap(const boost::shared_ptr<CostDataSum>& data, const Eigen::VectorXd& x, const bool& recalc) {
    calcDiff(data, x, unone_, recalc);
  }

#endif
};

struct CostDataSum {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataSum(Model* const model, pinocchio::Data* const data) : pinocchio(data), cost(0.) {
    for (CostModelSum::CostModelContainer::const_iterator it = model->get_costs().begin();
         it != model->get_costs().end(); ++it) {
      const CostItem& item = it->second;
      costs.insert(std::make_pair(item.name, item.cost->createData(data)));
    }
    const int& ndx = model->get_state().get_ndx();
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

  CostModelSum::CostDataContainer costs;
  pinocchio::Data* pinocchio;
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

#endif  // CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
