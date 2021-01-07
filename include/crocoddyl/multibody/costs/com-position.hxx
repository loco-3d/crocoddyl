///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const Vector3s& cref, const std::size_t& nu)
    : Base(state, activation, boost::make_shared<ResidualModelCoMPosition>(state, cref, nu)), cref_(cref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const Vector3s& cref)
    : Base(state, activation, boost::make_shared<ResidualModelCoMPosition>(state, cref)), cref_(cref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref,
                                                         const std::size_t& nu)
    : Base(state, boost::make_shared<ResidualModelCoMPosition>(state, cref, nu)), cref_(cref) {}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref)
    : Base(state, boost::make_shared<ResidualModelCoMPosition>(state, cref)), cref_(cref) {}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::~CostModelCoMPositionTpl() {}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual give the reference CoMPosition position
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x,
                                               const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the CoM tracking
  const std::size_t nv = state_->get_nv();
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);
  Eigen::Block<MatrixXs, -1, -1, true> Jcom = d->residual->Rx.leftCols(nv);
  data->Lx.head(nv).noalias() = Jcom.transpose() * data->activation->Ar;
  d->Arr_Jcom.noalias() = data->activation->Arr * Jcom;
  data->Lxx.topLeftCorner(nv, nv).noalias() = Jcom.transpose() * d->Arr_Jcom;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelCoMPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(Vector3s)) {
    cref_ = *static_cast<const Vector3s*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector3s)");
  }
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(Vector3s)) {
    Eigen::Map<Vector3s> ref_map(static_cast<Vector3s*>(pv)->data());
    ref_map[0] = cref_[0];
    ref_map[1] = cref_[1];
    ref_map[2] = cref_[2];
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector3s)");
  }
}

}  // namespace crocoddyl
