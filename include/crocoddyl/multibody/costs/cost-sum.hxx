///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelSumTpl<Scalar>::CostModelSumTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu,
                                         const bool& with_residuals)
    : state_(state), nu_(nu), nr_(0), with_residuals_(with_residuals) {}

template <typename Scalar>
CostModelSumTpl<Scalar>::CostModelSumTpl(boost::shared_ptr<StateMultibody> state, const bool& with_residuals)
    : state_(state), nu_(state->get_nv()), nr_(0), with_residuals_(with_residuals) {}

template <typename Scalar>
CostModelSumTpl<Scalar>::~CostModelSumTpl() {}

template <typename Scalar>
void CostModelSumTpl<Scalar>::addCost(const std::string& name, boost::shared_ptr<CostModelAbstract> cost,
                                      const Scalar& weight) {
  if (cost->get_nu() != nu_) {
    throw_pretty("Cost item doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  std::pair<typename CostModelContainer::iterator, bool> ret =
      costs_.insert(std::make_pair(name, CostItem(name, cost, weight)));
  if (ret.second == false) {
    std::cout << "Warning: this cost item already existed, we cannot add it" << std::endl;
  } else {
    nr_ += cost->get_activation()->get_nr();
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::removeCost(const std::string& name) {
  typename CostModelContainer::iterator it = costs_.find(name);
  if (it != costs_.end()) {
    nr_ -= it->second.cost->get_activation()->get_nr();
    costs_.erase(it);
  } else {
    std::cout << "Warning: this cost item doesn't exist, we cannot remove it" << std::endl;
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calc(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                   const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  data->cost = 0.;
  std::size_t nr = 0;

  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(), end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const CostItem& m_i = it_m->second;
    boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the cost name between data and model");

    m_i.cost->calc(d_i, x, u);
    data->cost += m_i.weight * d_i->cost;
    if (with_residuals_) {
      const std::size_t& nr_i = m_i.cost->get_activation()->get_nr();
      data->r.segment(nr, nr_i) = sqrt(m_i.weight) * d_i->r;
      nr += nr_i;
    }
  }
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                       const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (data->costs.size() != costs_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of cost datas and models");
  }
  std::size_t nr = 0;
  data->Lx.setZero();
  data->Lu.setZero();
  data->Lxx.setZero();
  data->Lxu.setZero();
  data->Luu.setZero();

  const std::size_t& ndx = state_->get_ndx();
  typename CostModelContainer::iterator it_m, end_m;
  typename CostDataContainer::iterator it_d, end_d;
  for (it_m = costs_.begin(), end_m = costs_.end(), it_d = data->costs.begin(), end_d = data->costs.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const CostItem& m_i = it_m->second;
    boost::shared_ptr<CostDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the cost name between data and model");

    m_i.cost->calcDiff(d_i, x, u);
    data->Lx += m_i.weight * d_i->Lx;
    data->Lu += m_i.weight * d_i->Lu;
    data->Lxx += m_i.weight * d_i->Lxx;
    data->Lxu += m_i.weight * d_i->Lxu;
    data->Luu += m_i.weight * d_i->Luu;
    if (with_residuals_) {
      const std::size_t& nr_i = m_i.cost->get_activation()->get_nr();
      data->Rx.block(nr, 0, nr_i, ndx) = sqrt(m_i.weight) * d_i->Rx;
      data->Ru.block(nr, 0, nr_i, nu_) = sqrt(m_i.weight) * d_i->Ru;
      nr += nr_i;
    }
  }
}

template <typename Scalar>
boost::shared_ptr<CostDataSumTpl<Scalar> > CostModelSumTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataSumTpl<Scalar> >(this, data);
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calc(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                   const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void CostModelSumTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataSumTpl<Scalar> >& data,
                                       const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& CostModelSumTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename CostModelSumTpl<Scalar>::CostModelContainer& CostModelSumTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::size_t& CostModelSumTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
const std::size_t& CostModelSumTpl<Scalar>::get_nr() const {
  return nr_;
}

}  // namespace crocoddyl
