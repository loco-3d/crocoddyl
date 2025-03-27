///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelFramePlacementTpl<Scalar>::ResidualModelFramePlacementTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const std::size_t nu)
    : Base(state, 6, nu, true, false, false),
      id_(id),
      pref_(pref),
      oMf_inv_(pref.inverse()),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFramePlacementTpl<Scalar>::ResidualModelFramePlacementTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref)
    : Base(state, 6, true, false, false),
      id_(id),
      pref_(pref),
      oMf_inv_(pref.inverse()),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rMf = oMf_inv_ * d->pinocchio->oMf[id_];
  data->r = pinocchio::log6(d->rMf).toVector();
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);
  data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelFramePlacementTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelFramePlacementTpl<NewScalar>
ResidualModelFramePlacementTpl<Scalar>::cast() const {
  typedef ResidualModelFramePlacementTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, pref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  typename SE3::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, pref_.rotation());
  os << "ResidualModelFramePlacement {frame=" << pin_model_->frames[id_].name
     << ", tref=" << pref_.translation().transpose().format(fmt)
     << ", qref=" << qref.coeffs().transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFramePlacementTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::SE3Tpl<Scalar>&
ResidualModelFramePlacementTpl<Scalar>::get_reference() const {
  return pref_;
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFramePlacementTpl<Scalar>::set_reference(
    const SE3& placement) {
  pref_ = placement;
  oMf_inv_ = placement.inverse();
}

}  // namespace crocoddyl
