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
#include "crocoddyl/core/diff-action-base.hpp"
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
  unsigned int const& get_nu() const;
  unsigned int const& get_nr() const;

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
  CostDataSum(Model* const model, pinocchio::Data* const data)
      : r_internal(model->get_nr()),
        Lx_internal(model->get_state().get_ndx()),
        Lu_internal(model->get_nu()),
        Lxx_internal(model->get_state().get_ndx(), model->get_state().get_ndx()),
        Lxu_internal(model->get_state().get_ndx(), model->get_nu()),
        Luu_internal(model->get_nu(), model->get_nu()),
        pinocchio(data),
        cost(0.),
        r(&r_internal(0), r_internal.size()),
        Lx(&Lx_internal(0), Lx_internal.size()),
        Lu(&Lu_internal(0), Lu_internal.size()),
        Lxx(&Lxx_internal(0), Lxx_internal.rows(), Lxx_internal.cols()),
        Lxu(&Lxu_internal(0), Lxu_internal.rows(), Lxu_internal.cols()),
        Luu(&Luu_internal(0), Luu_internal.rows(), Luu_internal.cols()),
        Rx(model->get_nr(), model->get_state().get_ndx()),
        Ru(model->get_nr(), model->get_nu()) {
    r.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
    for (CostModelSum::CostModelContainer::const_iterator it = model->get_costs().begin();
         it != model->get_costs().end(); ++it) {
      const CostItem& item = it->second;
      costs.insert(std::make_pair(item.name, item.cost->createData(data)));
    }
  }

  void shareMemory(DifferentialActionDataAbstract* const model) {
    // Share memory with the differential action data
    new (&r) Eigen::Map<Eigen::VectorXd>(&model->r(0), model->r.size());
    new (&Lx) Eigen::Map<Eigen::VectorXd>(&model->Lx(0), model->Lx.size());
    new (&Lu) Eigen::Map<Eigen::VectorXd>(&model->Lu(0), model->Lu.size());
    new (&Lxx) Eigen::Map<Eigen::MatrixXd>(&model->Lxx(0), model->Lxx.rows(), model->Lxx.cols());
    new (&Lxu) Eigen::Map<Eigen::MatrixXd>(&model->Lxu(0), model->Lxu.rows(), model->Lxu.cols());
    new (&Luu) Eigen::Map<Eigen::MatrixXd>(&model->Luu(0), model->Luu.rows(), model->Luu.cols());
  }

  Eigen::VectorXd get_r() const { return r; }
  Eigen::VectorXd get_Lx() const { return Lx; }
  Eigen::VectorXd get_Lu() const { return Lu; }
  Eigen::MatrixXd get_Lxx() const { return Lxx; }
  Eigen::MatrixXd get_Lxu() const { return Lxu; }
  Eigen::MatrixXd get_Luu() const { return Luu; }

  void set_r(Eigen::VectorXd _r) { r = _r; }
  void set_Lx(Eigen::VectorXd _Lx) { Lx = _Lx; }
  void set_Lu(Eigen::VectorXd _Lu) { Lu = _Lu; }
  void set_Lxx(Eigen::MatrixXd _Lxx) { Lxx = _Lxx; }
  void set_Lxu(Eigen::MatrixXd _Lxu) { Lxu = _Lxu; }
  void set_Luu(Eigen::MatrixXd _Luu) { Luu = _Luu; }

  // Creates internal data in case we don't share it externally
  Eigen::VectorXd r_internal;
  Eigen::VectorXd Lx_internal;
  Eigen::VectorXd Lu_internal;
  Eigen::MatrixXd Lxx_internal;
  Eigen::MatrixXd Lxu_internal;
  Eigen::MatrixXd Luu_internal;

  CostModelSum::CostDataContainer costs;
  pinocchio::Data* pinocchio;
  double cost;
  Eigen::Map<Eigen::VectorXd> r;
  Eigen::Map<Eigen::VectorXd> Lx;
  Eigen::Map<Eigen::VectorXd> Lu;
  Eigen::Map<Eigen::MatrixXd> Lxx;
  Eigen::Map<Eigen::MatrixXd> Lxu;
  Eigen::Map<Eigen::MatrixXd> Luu;
  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
