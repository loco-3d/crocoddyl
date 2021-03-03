///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/contact-force.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrameForce& fref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, nu)),
      fref_(fref) {
  if (activation_->get_nr() > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrameForce& fref)
    : Base(state, activation, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force)), fref_(fref) {
  if (activation_->get_nr() > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrameForce& fref, const std::size_t,
                                                           const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force, nu)), fref_(fref) {}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrameForce& fref, const std::size_t)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force)), fref_(fref) {}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrameForce& fref)
    : Base(state, boost::make_shared<ResidualModelContactForce>(state, fref.id, fref.force)), fref_(fref) {}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::~CostModelContactForceTpl() {}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference force
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>& x,
                                                const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and force residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->residual->Ru.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  d->Arr_Ru.noalias() = data->activation->Arr * data->residual->Ru;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
  data->Lxu.noalias() = data->residual->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactForceTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(FrameForce)) {
    FrameForce& ref_map = *static_cast<FrameForce*>(pv);
    ResidualModelContactForce* residual = static_cast<ResidualModelContactForce*>(residual_.get());
    fref_.id = residual->get_id();
    fref_.force = residual->get_reference();
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameForce)) {
    fref_ = *static_cast<const FrameForce*>(pv);
    ResidualModelContactForce* residual = static_cast<ResidualModelContactForce*>(residual_.get());
    residual->set_id(fref_.id);
    residual->set_reference(fref_.force);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

}  // namespace crocoddyl
