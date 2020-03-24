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

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct CostItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef CostModelAbstractTpl<Scalar> CostModelAbstract;

  CostItemTpl() {}
  CostItemTpl(const std::string& name, boost::shared_ptr<CostModelAbstract> cost, const Scalar& weight,
              bool active = true)
      : name(name), cost(cost), weight(weight), active(active) {}

  std::string name;
  boost::shared_ptr<CostModelAbstract> cost;
  Scalar weight;
  bool active;
};

template <typename _Scalar>
class CostModelSumTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef CostItemTpl<Scalar> CostItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, boost::shared_ptr<CostItem> > CostModelContainer;
  typedef std::map<std::string, boost::shared_ptr<CostDataAbstract> > CostDataContainer;

  CostModelSumTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  explicit CostModelSumTpl(boost::shared_ptr<StateMultibody> state);
  ~CostModelSumTpl();

  void addCost(const std::string& name, boost::shared_ptr<CostModelAbstract> cost, const Scalar& weight,
               bool active = true);
  void removeCost(const std::string& name);
  void changeCostStatus(const std::string& name, bool active);

  void calc(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataSumTpl<Scalar> > createData(DataCollectorAbstract* const data);

  void calc(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const Eigen::Ref<const VectorXs>& x);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const CostModelContainer& get_costs() const;
  const std::size_t& get_nu() const;
  const std::size_t& get_nr() const;

 private:
  boost::shared_ptr<StateMultibody> state_;
  CostModelContainer costs_;
  std::size_t nu_;
  std::size_t nr_;
  VectorXs unone_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const VectorXs& x,
                 const VectorXs& u = VectorXs()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
  }

  void calcDiff_wrap(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const VectorXs& x, const VectorXs& u) {
    calcDiff(data, x, u);
  }
  void calcDiff_wrap(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data, const VectorXs& x) {
    calcDiff(data, x, unone_);
  }

#endif
};

template <typename _Scalar>
struct CostDataSumTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef CostItemTpl<Scalar> CostItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataSumTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Lx_internal(model->get_state()->get_ndx()),
        Lu_internal(model->get_nu()),
        Lxx_internal(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu_internal(model->get_state()->get_ndx(), model->get_nu()),
        Luu_internal(model->get_nu(), model->get_nu()),
        shared(data),
        cost(0.),
        Lx(Lx_internal.data(), model->get_state()->get_ndx()),
        Lu(Lu_internal.data(), model->get_nu()),
        Lxx(Lxx_internal.data(), model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(Lxu_internal.data(), model->get_state()->get_ndx(), model->get_nu()),
        Luu(Luu_internal.data(), model->get_nu(), model->get_nu()) {
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
    for (typename CostModelSumTpl<Scalar>::CostModelContainer::const_iterator it = model->get_costs().begin();
         it != model->get_costs().end(); ++it) {
      const boost::shared_ptr<CostItem>& item = it->second;
      costs.insert(std::make_pair(item->name, item->cost->createData(data)));
    }
  }

  template <class ActionData>
  void shareMemory(ActionData* const data) {
    // Share memory with the differential action data
    new (&Lx) Eigen::Map<VectorXs>(data->Lx.data(), data->Lx.size());
    new (&Lu) Eigen::Map<VectorXs>(data->Lu.data(), data->Lu.size());
    new (&Lxx) Eigen::Map<MatrixXs>(data->Lxx.data(), data->Lxx.rows(), data->Lxx.cols());
    new (&Lxu) Eigen::Map<MatrixXs>(data->Lxu.data(), data->Lxu.rows(), data->Lxu.cols());
    new (&Luu) Eigen::Map<MatrixXs>(data->Luu.data(), data->Luu.rows(), data->Luu.cols());
  }

  VectorXs get_Lx() const { return Lx; }
  VectorXs get_Lu() const { return Lu; }
  MatrixXs get_Lxx() const { return Lxx; }
  MatrixXs get_Lxu() const { return Lxu; }
  MatrixXs get_Luu() const { return Luu; }

  void set_Lx(const VectorXs& _Lx) {
    if (Lx.size() != _Lx.size()) {
      throw_pretty("Invalid argument: "
                   << "Lx has wrong dimension (it should be " + std::to_string(Lx.size()) + ")");
    }
    Lx = _Lx;
  }
  void set_Lu(const VectorXs& _Lu) {
    if (Lu.size() != _Lu.size()) {
      throw_pretty("Invalid argument: "
                   << "Lu has wrong dimension (it should be " + std::to_string(Lu.size()) + ")");
    }
    Lu = _Lu;
  }
  void set_Lxx(const MatrixXs& _Lxx) {
    if (Lxx.rows() != _Lxx.rows() || Lxx.cols() != _Lxx.cols()) {
      throw_pretty("Invalid argument: "
                   << "Lxx has wrong dimension (it should be " + std::to_string(Lxx.rows()) + ", " +
                          std::to_string(Lxx.cols()) + ")");
    }
    Lxx = _Lxx;
  }
  void set_Lxu(const MatrixXs& _Lxu) {
    if (Lxu.rows() != _Lxu.rows() || Lxu.cols() != _Lxu.cols()) {
      throw_pretty("Invalid argument: "
                   << "Lxu has wrong dimension (it should be " + std::to_string(Lxu.rows()) + ", " +
                          std::to_string(Lxu.cols()) + ")");
    }
    Lxu = _Lxu;
  }
  void set_Luu(const MatrixXs& _Luu) {
    if (Luu.rows() != _Luu.rows() || Luu.cols() != _Luu.cols()) {
      throw_pretty("Invalid argument: "
                   << "Luu has wrong dimension (it should be " + std::to_string(Luu.rows()) + ", " +
                          std::to_string(Luu.cols()) + ")");
    }
    Luu = _Luu;
  }

  // Creates internal data in case we don't share it externally
  VectorXs Lx_internal;
  VectorXs Lu_internal;
  MatrixXs Lxx_internal;
  MatrixXs Lxu_internal;
  MatrixXs Luu_internal;

  typename CostModelSumTpl<Scalar>::CostDataContainer costs;
  DataCollectorAbstract* shared;
  Scalar cost;
  Eigen::Map<VectorXs> Lx;
  Eigen::Map<VectorXs> Lu;
  Eigen::Map<MatrixXs> Lxx;
  Eigen::Map<MatrixXs> Lxu;
  Eigen::Map<MatrixXs> Luu;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/cost-sum.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
