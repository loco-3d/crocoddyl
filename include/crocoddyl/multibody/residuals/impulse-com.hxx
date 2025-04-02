///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelImpulseCoMTpl<Scalar>::ResidualModelImpulseCoMTpl(
    std::shared_ptr<StateMultibody> state)
    : Base(state, 3, 0, true, true, false),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
void ResidualModelImpulseCoMTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  // Compute the residual residual give the reference CoM position
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);

  pinocchio::centerOfMass(*pin_model_.get(), d->pinocchio_internal, q,
                          d->impulses->vnext - v);
  data->r = d->pinocchio_internal.vcom[0];
}

template <typename Scalar>
void ResidualModelImpulseCoMTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  const std::size_t ndx = state_->get_ndx();
  pinocchio::getCenterOfMassVelocityDerivatives(
      *pin_model_.get(), d->pinocchio_internal, d->dvc_dq);
  pinocchio::jacobianCenterOfMass(*pin_model_.get(), d->pinocchio_internal,
                                  false);
  d->ddv_dv = d->impulses->dvnext_dx.rightCols(ndx - nv);
  d->ddv_dv.diagonal().array() -= Scalar(1);
  data->Rx.leftCols(nv) = d->dvc_dq;
  data->Rx.leftCols(nv).noalias() +=
      d->pinocchio_internal.Jcom * d->impulses->dvnext_dx.leftCols(nv);
  data->Rx.rightCols(ndx - nv).noalias() =
      d->pinocchio_internal.Jcom * d->ddv_dv;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelImpulseCoMTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelImpulseCoMTpl<NewScalar> ResidualModelImpulseCoMTpl<Scalar>::cast()
    const {
  typedef ResidualModelImpulseCoMTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()));
  return ret;
}

template <typename Scalar>
void ResidualModelImpulseCoMTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelImpulseCoM";
}

}  // namespace crocoddyl
