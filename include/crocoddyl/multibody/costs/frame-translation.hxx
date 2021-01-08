///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameTranslation& xref, const std::size_t& nu)
    : Base(state, activation, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation, nu)),
      xref_(xref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameTranslation& xref)
    : Base(state, activation, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation)),
      xref_(xref) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const FrameTranslation& xref, const std::size_t& nu)
    : Base(state, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation, nu)),
      xref_(xref) {}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                   const FrameTranslation& xref)
    : Base(state, boost::make_shared<ResidualModelFrameTranslation>(state, xref.id, xref.translation)), xref_(xref) {}

template <typename Scalar>
CostModelFrameTranslationTpl<Scalar>::~CostModelFrameTranslationTpl() {}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>& x,
                                                const Eigen::Ref<const VectorXs>& u) {
  // Compute the frame translation w.r.t. the reference frame
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                    const Eigen::Ref<const VectorXs>& x,
                                                    const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and frame translation residual models
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(d->activation, d->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  Eigen::Ref<Matrix3xs> J(data->residual->Rx.leftCols(nv));
  d->Lx.head(nv) = J.transpose() * d->activation->Ar;
  d->Arr_J.noalias() = data->activation->Arr * J;
  d->Lxx.topLeftCorner(nv, nv) = J.transpose() * d->Arr_J;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFrameTranslationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameTranslation)) {
    FrameTranslation& ref_map = *static_cast<FrameTranslation*>(pv);
    ref_map = xref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameTranslation)");
  }
}

template <typename Scalar>
void CostModelFrameTranslationTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameTranslation)) {
    xref_ = *static_cast<const FrameTranslation*>(pv);
    ResidualModelFrameTranslation* residual = static_cast<ResidualModelFrameTranslation*>(residual_.get());
    residual->set_id(xref_.id);
    residual->set_reference(xref_.translation);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameTranslation)");
  }
}

}  // namespace crocoddyl
