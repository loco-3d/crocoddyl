///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
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
    const Vector6s& href, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelCentroidalMomentum>(state, href, nu)), href_(href) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const Vector6s& href)
    : Base(state, activation, boost::make_shared<ResidualModelCentroidalMomentum>(state, href)), href_(href) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const Vector6s& href, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelCentroidalMomentum>(state, href, nu)), href_(href) {}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const Vector6s& href)
    : Base(state, boost::make_shared<ResidualModelCentroidalMomentum>(state, href)), href_(href) {}

template <typename Scalar>
CostModelCentroidalMomentumTpl<Scalar>::~CostModelCentroidalMomentumTpl() {}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>& x,
                                                  const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference centroidal momentum
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>& x,
                                                      const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and centroidal momentum residual models
  Data* d = static_cast<Data*>(data.get());
  activation_->calcDiff(data->activation, data->residual->r);
  residual_->calcDiff(data->residual, x, u);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  Eigen::Ref<Matrix6xs> Rx(data->residual->Rx);
  d->Arr_Rx.noalias() = data->activation->Arr * Rx;
  data->Lx.noalias() = Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelCentroidalMomentumTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(Vector6s)) {
    href_ = *static_cast<const Vector6s*>(pv);
    ResidualModelCentroidalMomentum* residual = static_cast<ResidualModelCentroidalMomentum*>(residual_.get());
    residual->set_reference(href_);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector6s)");
  }
}

template <typename Scalar>
void CostModelCentroidalMomentumTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
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

}  // namespace crocoddyl
