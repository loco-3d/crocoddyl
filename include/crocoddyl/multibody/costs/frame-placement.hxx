///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"

#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FramePlacement& Mref, const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement, nu)),
      Mref_(Mref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                                               const FramePlacement& Mref)
    : Base(state, activation, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement)),
      Mref_(Mref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FramePlacement& Mref, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement, nu)), Mref_(Mref) {}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                                                               const FramePlacement& Mref)
    : Base(state, boost::make_shared<ResidualModelFramePlacement>(state, Mref.id, Mref.placement)), Mref_(Mref) {}

template <typename Scalar>
CostModelFramePlacementTpl<Scalar>::~CostModelFramePlacementTpl() {}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x,
                                              const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the reference frame placement
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>& x,
                                                  const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and frame placement residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  const std::size_t nv = state_->get_nv();
  Eigen::Ref<Matrix6xs> J(data->residual->Rx.leftCols(nv));
  data->Lx.head(nv).noalias() = J.transpose() * data->activation->Ar;
  d->Arr_J.noalias() = data->activation->Arr * J;
  data->Lxx.topLeftCorner(nv, nv).noalias() = J.transpose() * d->Arr_J;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelFramePlacementTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(FramePlacement)) {
    FramePlacement& ref_map = *static_cast<FramePlacement*>(pv);
    ref_map = Mref_;
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FramePlacement)");
  }
}

template <typename Scalar>
void CostModelFramePlacementTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(FramePlacement)) {
    Mref_ = *static_cast<const FramePlacement*>(pv);
    ResidualModelFramePlacement* residual = static_cast<ResidualModelFramePlacement*>(residual_.get());
    residual->set_id(Mref_.id);
    residual->set_reference(Mref_.placement);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be FramePlacement)");
  }
}

}  // namespace crocoddyl
