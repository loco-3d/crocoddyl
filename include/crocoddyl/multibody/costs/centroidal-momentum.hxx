///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/centroidal-momentum.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const Vector6s& href, const std::size_t& nu)
    : Base(state, activation, nu), href_(href) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const Vector6s& href)
    : Base(state, activation), href_(href) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const Vector6s& href, const std::size_t& nu)
    : Base(state, 6, nu), href_(href) {}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const Vector6s& href)
    : Base(state, 6), href_(href) {}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::~CostModelCentroidalMomentumTpl() {}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  // Compute the cost residual give the reference CentroidalMomentum
  CostDataCentroidalMomentumTpl<Scalar>* d = static_cast<CostDataCentroidalMomentumTpl<Scalar>*>(data.get());
  data->r = d->pinocchio->hg.toVector() - href_;

  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  CostDataCentroidalMomentumTpl<Scalar>* d = static_cast<CostDataCentroidalMomentumTpl<Scalar>*>(data.get());
  const std::size_t& nv = state_->get_nv();
  Eigen::Ref<Matrix6xs> Rq = data->Rx.leftCols(nv);
  Eigen::Ref<Matrix6xs> Rv = data->Rx.rightCols(nv);

  activation_->calcDiff(data->activation, data->r);
  pinocchio::getCentroidalDynamicsDerivatives(*state_->get_pinocchio().get(), *d->pinocchio, Rq, d->dhd_dq, d->dhd_dv,
                                              Rv);

  // The derivative computation in pinocchio does not take the frame of reference into
  // account. So we need to update the com frame as well.
  for (int i = 0; i < d->pinocchio->Jcom.cols(); ++i) {
    data->Rx.template block<3, 1>(3, i) -= d->pinocchio->Jcom.col(i).cross(d->pinocchio->hg.linear());
  }

  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelCentroidalMomentumTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataCentroidalMomentumTpl<Scalar> >(this, data);
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(Vector6s)) {
    href_ = *static_cast<const Vector6s*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector6s)");
  }
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(Vector6s)) {
    Eigen::Map<Vector6s> ref_map(static_cast<Vector6s*>(pv)->data());
    ref_map[0] = href_[0];
    ref_map[1] = href_[1];
    ref_map[2] = href_[2];
    ref_map[3] = href_[3];
    ref_map[4] = href_[4];
    ref_map[5] = href_[5];
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector6s)");
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector6s& CostModelCentroidalMomentumTpl<Scalar>::get_href() const {
  return href_;
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::set_href(const Vector6s& href_in) {
  href_ = href_in;
}

}  // namespace crocoddyl
