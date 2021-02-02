///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameRotation& Rref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation, nu)),
      Rref_(Rref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                                             const FrameRotation& Rref)
    : Base(state, activation, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation)),
      Rref_(Rref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameRotation& Rref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation, nu)), Rref_(Rref) {}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                                                             const FrameRotation& Rref)
    : Base(state, boost::make_shared<ResidualModelFrameRotation>(state, Rref.id, Rref.rotation)), Rref_(Rref) {}

template <typename Scalar>
CostModelFrameRotationTpl<Scalar>::~CostModelFrameRotationTpl() {}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& x,
                                             const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference frame rotation
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(d->activation, d->residual->r);
  d->cost = d->activation->a_value;
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and frame placement residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(d->activation, d->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  const std::size_t nv = state_->get_nv();
  Eigen::Ref<Matrix3xs> J(data->residual->Rx.leftCols(nv));
  data->Lx.head(nv).noalias() = J.transpose() * data->activation->Ar;
  d->Arr_J.noalias() = data->activation->Arr * J;
  data->Lxx.topLeftCorner(nv, nv).noalias() = J.transpose() * d->Arr_J;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameRotationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameRotation)) {
    FrameRotation& ref_map = *static_cast<FrameRotation*>(pv);
    ref_map = Rref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameRotation)");
  }
}

template <typename Scalar>
void CostModelFrameRotationTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameRotation)) {
    Rref_ = *static_cast<const FrameRotation*>(pv);
    ResidualModelFrameRotation* residual = static_cast<ResidualModelFrameRotation*>(residual_.get());
    residual->set_id(Rref_.id);
    residual->set_reference(Rref_.rotation);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameRotation)");
  }
}

}  // namespace crocoddyl
