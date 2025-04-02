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
ResidualModelFrameVelocityTpl<Scalar>::ResidualModelFrameVelocityTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Motion& vref, const pinocchio::ReferenceFrame type,
    const std::size_t nu)
    : Base(state, 6, nu, true, true, false),
      id_(id),
      vref_(vref),
      type_(type),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelFrameVelocityTpl<Scalar>::ResidualModelFrameVelocityTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Motion& vref, const pinocchio::ReferenceFrame type)
    : Base(state, 6, true, true, false),
      id_(id),
      vref_(vref),
      type_(type),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  data->r = (pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_,
                                         type_) -
             vref_)
                .toVector();
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Get the partial derivatives of the local frame velocity
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio, id_,
                                         type_, data->Rx.leftCols(nv),
                                         data->Rx.rightCols(nv));
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelFrameVelocityTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelFrameVelocityTpl<NewScalar>
ResidualModelFrameVelocityTpl<Scalar>::cast() const {
  typedef ResidualModelFrameVelocityTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, vref_.template cast<NewScalar>(), type_, nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelFrameVelocity {frame=" << pin_model_->frames[id_].name
     << ", vref=" << vref_.toVector().transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameVelocityTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::MotionTpl<Scalar>&
ResidualModelFrameVelocityTpl<Scalar>::get_reference() const {
  return vref_;
}

template <typename Scalar>
pinocchio::ReferenceFrame ResidualModelFrameVelocityTpl<Scalar>::get_type()
    const {
  return type_;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::set_reference(
    const Motion& velocity) {
  vref_ = velocity;
}

template <typename Scalar>
void ResidualModelFrameVelocityTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  type_ = type;
}

}  // namespace crocoddyl
