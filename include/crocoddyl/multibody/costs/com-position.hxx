///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation, const Vector3s &cref,
    const std::size_t &nu)
    : Base(state, activation, nu), cref_(cref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation, const Vector3s &cref)
    : Base(state, activation), cref_(cref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(
    boost::shared_ptr<StateMultibody> state, const Vector3s &cref,
    const std::size_t &nu)
    : Base(state, 3, nu), cref_(cref) {}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::CostModelCoMPositionTpl(
    boost::shared_ptr<StateMultibody> state, const Vector3s &cref)
    : Base(state, 3), cref_(cref) {}

template <typename Scalar>
CostModelCoMPositionTpl<Scalar>::~CostModelCoMPositionTpl() {}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  // Compute the cost residual give the reference CoMPosition position
  Data *d = static_cast<Data *>(data.get());
  data->r = d->pinocchio->com[0] - cref_;

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t &nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r);
  data->Rx.leftCols(nv) = d->pinocchio->Jcom;
  data->Lx.head(nv).noalias() =
      d->pinocchio->Jcom.transpose() * data->activation->Ar;
  d->Arr_Jcom.noalias() = data->activation->Arr * d->pinocchio->Jcom;
  data->Lxx.topLeftCorner(nv, nv).noalias() =
      d->pinocchio->Jcom.transpose() * d->Arr_Jcom;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelCoMPositionTpl<Scalar>::createData(DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::set_referenceImpl(
    const std::type_info &ti, const void *pv) {
  if (ti == typeid(Vector3s)) {
    cref_ = *static_cast<const Vector3s *>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector3s)");
  }
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::get_referenceImpl(
    const std::type_info &ti, void *pv) const {
  if (ti == typeid(Vector3s)) {
    Eigen::Map<Vector3s> ref_map(static_cast<Vector3s *>(pv)->data());
    ref_map[0] = cref_[0];
    ref_map[1] = cref_[1];
    ref_map[2] = cref_[2];
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be Vector3s)");
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s &
CostModelCoMPositionTpl<Scalar>::get_cref() const {
  return cref_;
}

template <typename Scalar>
void CostModelCoMPositionTpl<Scalar>::set_cref(const Vector3s &cref_in) {
  cref_ = cref_in;
}

} // namespace crocoddyl
