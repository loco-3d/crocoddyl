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

class CostModelSum : public CostModelAbstract {
 public:
  typedef std::map<std::string, CostItem> CostModelContainer;
  typedef std::map<std::string, boost::shared_ptr<CostDataAbstract> > CostDataContainer;

  CostModelSum(StateMultibody& state, unsigned int const& nu, const bool& with_residuals = true);
  explicit CostModelSum(StateMultibody& state, const bool& with_residuals = true);
  ~CostModelSum();

  void addCost(const std::string& name, CostModelAbstract* const cost, const double& weight);
  void removeCost(const std::string& name);

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const CostModelContainer& get_costs() const;
  const unsigned int& get_nr() const;

 private:
  CostModelContainer costs_;
  unsigned int nr_;
};

struct CostDataSum : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataSum(Model* const model, pinocchio::Data* const data) : CostDataAbstract(model, data) {
    for (CostModelSum::CostModelContainer::const_iterator it = model->get_costs().begin();
         it != model->get_costs().end(); ++it) {
      const CostItem& item = it->second;
      costs.insert(std::make_pair(item.name, item.cost->createData(data)));
    }
  }

  CostModelSum::CostDataContainer costs;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
