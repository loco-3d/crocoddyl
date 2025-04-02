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
ResidualModelFrameRotationTpl<Scalar>::ResidualModelFrameRotationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref, const std::size_t nu)
    : Base(state, 3, nu, true, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFrameRotationTpl<Scalar>::ResidualModelFrameRotationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref)
    : Base(state, 3, true, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame rotation w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rRf.noalias() = oRf_inv_ * d->pinocchio->oMf[id_].rotation();
  data->r = pinocchio::log3(d->rRf);
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog3(d->rRf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);

  // Compute the derivatives of the frame rotation
  const std::size_t nv = state_->get_nv();
  data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf.template bottomRows<3>();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelFrameRotationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelFrameRotationTpl<NewScalar>
ResidualModelFrameRotationTpl<Scalar>::cast() const {
  typedef ResidualModelFrameRotationTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, Rref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  typename pinocchio::SE3Tpl<Scalar>::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, Rref_);
  os << "ResidualModelFrameRotation {frame=" << pin_model_->frames[id_].name
     << ", qref=" << qref.coeffs().transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameRotationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s&
ResidualModelFrameRotationTpl<Scalar>::get_reference() const {
  return Rref_;
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameRotationTpl<Scalar>::set_reference(
    const Matrix3s& rotation) {
  Rref_ = rotation;
  oRf_inv_ = rotation.transpose();
}

}  // namespace crocoddyl
