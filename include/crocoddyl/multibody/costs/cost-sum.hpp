///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
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
  CostItem(const std::string& name, boost::shared_ptr<CostModelAbstract> cost, const double& weight)
      : name(name), cost(cost), weight(weight) {}

  std::string name;
  boost::shared_ptr<CostModelAbstract> cost;
  double weight;
};

struct CostDataSum;  // forward declaration

class CostModelSum {
 public:
  typedef std::map<std::string, CostItem> CostModelContainer;
  typedef std::map<std::string, boost::shared_ptr<CostDataAbstract> > CostDataContainer;

  CostModelSum(boost::shared_ptr<StateMultibody> state, const std::size_t& nu, const bool& with_residuals = true);
  explicit CostModelSum(boost::shared_ptr<StateMultibody> state, const bool& with_residuals = true);
  ~CostModelSum();

  void addCost(const std::string& name, boost::shared_ptr<CostModelAbstract> cost, const double& weight);
  void removeCost(const std::string& name);

  void calc(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataSum> createData(pinocchio::Data* const data);

  void calc(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<CostDataSum>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const CostModelContainer& get_costs() const;
  const std::size_t& get_nu() const;
  const std::size_t& get_nr() const;

 private:
  boost::shared_ptr<StateMultibody> state_;
  CostModelContainer costs_;
  std::size_t nu_;
  std::size_t nr_;
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
        Lx_internal(model->get_state()->get_ndx()),
        Lu_internal(model->get_nu()),
        Lxx_internal(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu_internal(model->get_state()->get_ndx(), model->get_nu()),
        Luu_internal(model->get_nu(), model->get_nu()),
        pinocchio(data),
        cost(0.),
        r(r_internal.data(), model->get_nr()),
        Lx(Lx_internal.data(), model->get_state()->get_ndx()),
        Lu(Lu_internal.data(), model->get_nu()),
        Lxx(Lxx_internal.data(), model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(Lxu_internal.data(), model->get_state()->get_ndx(), model->get_nu()),
        Luu(Luu_internal.data(), model->get_nu(), model->get_nu()),
        Rx(model->get_nr(), model->get_state()->get_ndx()),
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

  template <typename ActionData>
  void shareMemory(ActionData* const data) {
    // Share memory with the differential action data
    new (&r) Eigen::Map<Eigen::VectorXd>(data->r.data(), data->r.size());
    new (&Lx) Eigen::Map<Eigen::VectorXd>(data->Lx.data(), data->Lx.size());
    new (&Lu) Eigen::Map<Eigen::VectorXd>(data->Lu.data(), data->Lu.size());
    new (&Lxx) Eigen::Map<Eigen::MatrixXd>(data->Lxx.data(), data->Lxx.rows(), data->Lxx.cols());
    new (&Lxu) Eigen::Map<Eigen::MatrixXd>(data->Lxu.data(), data->Lxu.rows(), data->Lxu.cols());
    new (&Luu) Eigen::Map<Eigen::MatrixXd>(data->Luu.data(), data->Luu.rows(), data->Luu.cols());
  }

  Eigen::VectorXd get_r() const { return r; }
  Eigen::VectorXd get_Lx() const { return Lx; }
  Eigen::VectorXd get_Lu() const { return Lu; }
  Eigen::MatrixXd get_Lxx() const { return Lxx; }
  Eigen::MatrixXd get_Lxu() const { return Lxu; }
  Eigen::MatrixXd get_Luu() const { return Luu; }

  void set_r(const Eigen::VectorXd& _r) {
    if (r.size() != _r.size()) {
      throw std::invalid_argument("r has wrong dimension (it should be " + std::to_string(r.size()) + ")");
    }
    r = _r;
  }
  void set_Lx(const Eigen::VectorXd& _Lx) {
    if (Lx.size() != _Lx.size()) {
      throw std::invalid_argument("Lx has wrong dimension (it should be " + std::to_string(Lx.size()) + ")");
    }
    Lx = _Lx;
  }
  void set_Lu(const Eigen::VectorXd& _Lu) {
    if (Lu.size() != _Lu.size()) {
      throw std::invalid_argument("Lu has wrong dimension (it should be " + std::to_string(Lu.size()) + ")");
    }
    Lu = _Lu;
  }
  void set_Lxx(const Eigen::MatrixXd& _Lxx) {
    if (Lxx.rows() != _Lxx.rows() && Lxx.cols() != _Lxx.cols()) {
      throw std::invalid_argument("Lxx has wrong dimension (it should be " + std::to_string(Lxx.rows()) + ", " +
                                  std::to_string(Lxx.cols()) + ")");
    }
    Lxx = _Lxx;
  }
  void set_Lxu(const Eigen::MatrixXd& _Lxu) {
    if (Lxu.rows() != _Lxu.rows() && Lxu.cols() != _Lxu.cols()) {
      throw std::invalid_argument("Lxu has wrong dimension (it should be " + std::to_string(Lxu.rows()) + ", " +
                                  std::to_string(Lxu.cols()) + ")");
    }
    Lxu = _Lxu;
  }
  void set_Luu(const Eigen::MatrixXd& _Luu) {
    if (Luu.rows() != _Luu.rows() && Luu.cols() != _Luu.cols()) {
      throw std::invalid_argument("Luu has wrong dimension (it should be " + std::to_string(Luu.rows()) + ", " +
                                  std::to_string(Luu.cols()) + ")");
    }
    Luu = _Luu;
  }

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
