///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrameForce& fref, const std::size_t& nu)
    : Base(state, activation, nu), fref_(fref) {
  if (activation_->get_nr() > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrameForce& fref)
    : Base(state, activation), fref_(fref) {
  if (activation_->get_nr() > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrameForce& fref, const std::size_t& nr,
                                                           const std::size_t& nu)
    : Base(state, nr, nu), fref_(fref) {
  if (nr > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrameForce& fref, const std::size_t& nr)
    : Base(state, nr), fref_(fref) {
  if (nr > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrameForce& fref)
    : Base(state, 6), fref_(fref) {}

template <typename Scalar>
CostModelContactForceTpl<Scalar>::~CostModelContactForceTpl() {}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& /*x*/,
                                            const Eigen::Ref<const VectorXs>& /*u*/) {
  Data* d = static_cast<Data*>(data.get());

  // We transform the force to the contact frame
  switch (d->contact_type) {
    case Contact3D:
      data->r = (d->contact->jMf.actInv(d->contact->f) - fref_.oFf).linear();
      break;
    case Contact6D:
      data->r = (d->contact->jMf.actInv(d->contact->f) - fref_.oFf).toVector();
      break;
    default:
      break;
  }

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;

  activation_->calcDiff(data->activation, data->r);
  switch (d->contact_type) {
    case Contact3D:
      data->Rx = df_dx.template topRows<3>();
      data->Ru = df_du.template topRows<3>();
      break;
    case Contact6D:
      data->Rx = df_dx;
      data->Ru = df_du;
      break;
    default:
      break;
  }
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->Ru.transpose() * data->activation->Ar;

  d->Arr_Ru.noalias() = data->activation->Arr * data->Ru;

  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
  data->Lxu.noalias() = data->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactForceTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FrameForce)) {
    fref_ = *static_cast<const FrameForce*>(pv);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FrameForce)) {
    FrameForce& ref_map = *static_cast<FrameForce*>(pv);
    ref_map = fref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FrameForce)");
  }
}

template <typename Scalar>
const FrameForceTpl<Scalar>& CostModelContactForceTpl<Scalar>::get_fref() const {
  return fref_;
}

template <typename Scalar>
void CostModelContactForceTpl<Scalar>::set_fref(const FrameForce& fref_in) {
  fref_ = fref_in;
}

}  // namespace crocoddyl
