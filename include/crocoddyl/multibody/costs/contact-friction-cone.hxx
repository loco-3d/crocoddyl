///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameFrictionCone &fref, const std::size_t &nu)
    : Base(state, activation, nu), fref_(fref) {
  if (activation_->get_nr() != fref_.cone.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation,
    const FrameFrictionCone &fref)
    : Base(state, activation), fref_(fref) {
  if (activation_->get_nr() != fref_.cone.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << fref_.cone.get_nf() + 1);
  }
}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, const FrameFrictionCone &fref,
    const std::size_t &nu)
    : Base(state, fref.cone.get_nf() + 1, nu), fref_(fref) {}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, const FrameFrictionCone &fref)
    : Base(state, fref.cone.get_nf() + 1), fref_(fref) {}

template <typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::~CostModelContactFrictionConeTpl() {}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  // Compute the residual of the friction cone. Note that we need to transform
  // the force to the contact frame
  data->r.noalias() =
      fref_.cone.get_A() * d->contact->jMf.actInv(d->contact->f).linear();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const MatrixXs &df_dx = d->contact->df_dx;
  const MatrixXs &df_du = d->contact->df_du;
  const MatrixX3s &A = fref_.cone.get_A();

  activation_->calcDiff(data->activation, data->r);
  if (d->more_than_3_constraints) {
    data->Rx.noalias() = A * df_dx.template topRows<3>();
    data->Ru.noalias() = A * df_du.template topRows<3>();
  } else {
    data->Rx.noalias() = A * df_dx;
    data->Ru.noalias() = A * df_du;
  }
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->Ru.transpose() * data->activation->Ar;

  d->Arr_Ru.noalias() = data->activation->Arr * data->Ru;
  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
  data->Lxu.noalias() = data->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelContactFrictionConeTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::set_referenceImpl(
    const std::type_info &ti, const void *pv) {
  if (ti == typeid(FrameFrictionCone)) {
    fref_ = *static_cast<const FrameFrictionCone *>(pv);
  } else {
    throw_pretty(
        "Invalid argument: incorrect type (it should be FrameFrictionCone)");
  }
}

template <typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::get_referenceImpl(
    const std::type_info &ti, void *pv) const {
  if (ti == typeid(FrameFrictionCone)) {
    FrameFrictionCone &ref_map = *static_cast<FrameFrictionCone *>(pv);
    ref_map = fref_;
  } else {
    throw_pretty(
        "Invalid argument: incorrect type (it should be FrameFrictionCone)");
  }
}

} // namespace crocoddyl
