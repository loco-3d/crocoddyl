///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/residuals/parameters.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelParametersTpl<Scalar>::ResidualModelParametersTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const VectorXs& pref,
    const std::size_t nu)
    : Base(state, static_cast<std::size_t>(pref.size()), nu,
           /*q_dependent=*/false, /*v_dependent=*/false,
           /*u_dependent=*/false, static_cast<std::size_t>(pref.size())),
      pref_(pref) {}

template <typename Scalar>
void ResidualModelParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  DataCollectorParams* d = dynamic_cast<DataCollectorParams*>(data->shared);
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must be DataCollectorParamsTpl");
  }
  if (d->params == nullptr) {
    throw_pretty("Invalid argument: parameter data is null");
  }
  if (static_cast<std::size_t>(d->params->p.size()) != np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be " +
                 std::to_string(np_) + ")");
  }
  data->r = d->params->p - pref_;
}

template <typename Scalar>
void ResidualModelParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelParametersTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>&,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Rp = I is constant and was set in createData — nothing to do.
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelParametersTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  DataCollectorParams* params = dynamic_cast<DataCollectorParams*>(data);
  if (params == nullptr || params->params == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide DataCollectorParamsTpl");
  }
  if (static_cast<std::size_t>(params->params->p.size()) != np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be "
                 << np_ << ")");
  }
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<ResidualDataAbstract>(
          Eigen::aligned_allocator<ResidualDataAbstract>(), this, data);
  // Rp = I  (nr_ == np_)
  d->Rp.diagonal().fill(Scalar(1.));
  return d;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelParametersTpl<NewScalar> ResidualModelParametersTpl<Scalar>::cast()
    const {
  typedef ResidualModelParametersTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 pref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelParametersTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelParameters {np=" << np_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ResidualModelParametersTpl<Scalar>::get_reference() const {
  return pref_;
}

template <typename Scalar>
void ResidualModelParametersTpl<Scalar>::set_reference(
    const VectorXs& reference) {
  if (static_cast<std::size_t>(reference.size()) != np_) {
    throw_pretty("Invalid argument: "
                 << "the parameter reference has wrong dimension ("
                 << reference.size()
                 << " provided - it should be " + std::to_string(np_) + ")");
  }
  pref_ = reference;
}

}  // namespace crocoddyl
