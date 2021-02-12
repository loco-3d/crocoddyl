///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/contact-impulse.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FrameForce& fref)
    : Base(state, activation, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, 0)),
      fref_(fref) {
  if (activation_->get_nr() > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FrameForce& fref, const std::size_t)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, 0)), fref_(fref) {}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FrameForce& fref)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, 0)), fref_(fref) {}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::~CostModelContactImpulseTpl() {}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x,
                                              const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference impulse
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(d->activation, d->residual->r);
  d->cost = d->activation->a_value;
}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>& x,
                                                  const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and force residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactImpulseTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameForce)) {
    fref_ = *static_cast<const FrameForce*>(pv);
    ResidualModelContactForce* residual = static_cast<ResidualModelContactForce*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.force);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameForce)) {
    FrameForce& ref_map = *static_cast<FrameForce*>(pv);
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

}  // namespace crocoddyl
