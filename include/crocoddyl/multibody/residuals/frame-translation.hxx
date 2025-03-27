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
ResidualModelFrameTranslationTpl<Scalar>::ResidualModelFrameTranslationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref, const std::size_t nu)
    : Base(state, 3, nu, true, false, false),
      id_(id),
      xref_(xref),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFrameTranslationTpl<Scalar>::ResidualModelFrameTranslationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref)
    : Base(state, 3, true, false, false),
      id_(id),
      xref_(xref),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Compute the frame translation w.r.t. the reference frame
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  data->r = d->pinocchio->oMf[id_].translation() - xref_;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame translation
  const std::size_t nv = state_->get_nv();
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);
  d->Rx.leftCols(nv).noalias() =
      d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
  ;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelFrameTranslationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelFrameTranslationTpl<NewScalar>
ResidualModelFrameTranslationTpl<Scalar>::cast() const {
  typedef ResidualModelFrameTranslationTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, xref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelFrameTranslation {frame=" << pin_model_->frames[id_].name
     << ", tref=" << xref_.transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameTranslationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
ResidualModelFrameTranslationTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::set_reference(
    const Vector3s& translation) {
  xref_ = translation;
}

}  // namespace crocoddyl
