///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-impulse.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FrameForce& fref)
    : Base(state, activation, 0), fref_(fref) {
  if (activation_->get_nr() > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FrameForce& fref, const std::size_t& nr)
    : Base(state, nr, 0), fref_(fref) {
  if (nr > 6) {
    throw_pretty("Invalid argument: "
                 << "nr is less than 6");
  }
}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FrameForce& fref)
    : Base(state, 6, 0), fref_(fref) {}

template <typename Scalar>
CostModelContactImpulseTpl<Scalar>::~CostModelContactImpulseTpl() {}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& /*x*/,
                                              const Eigen::Ref<const VectorXs>& /*u*/) {
  Data* d = static_cast<Data*>(data.get());

  // We transform the impulse to the contact frame
  switch (d->impulse_type) {
    case Impulse3D:
      data->r = (d->impulse->jMf.actInv(d->impulse->f) - fref_.force).linear();
      break;
    case Impulse6D:
      data->r = (d->impulse->jMf.actInv(d->impulse->f) - fref_.force).toVector();
      break;
    default:
      break;
  }

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->impulse->df_dx;

  activation_->calcDiff(data->activation, data->r);
  switch (d->impulse_type) {
    case Impulse3D:
      data->Rx = df_dx.template topRows<3>();
      break;
    case Impulse6D:
      data->Rx = df_dx;
      break;
    default:
      break;
  }
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
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

template <typename Scalar>
const FrameForceTpl<Scalar>& CostModelContactImpulseTpl<Scalar>::get_fref() const {
  return fref_;
}

template <typename Scalar>
void CostModelContactImpulseTpl<Scalar>::set_fref(const FrameForce& fref_in) {
  fref_ = fref_in;
}

}  // namespace crocoddyl
