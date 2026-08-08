///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelFramePlacementTpl<Scalar>::TaskModelFramePlacementTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const std::size_t nu)
    : TaskModelFramePlacementTpl(state, id, pref, pinocchio::LOCAL, nu) {}

template <typename Scalar>
TaskModelFramePlacementTpl<Scalar>::TaskModelFramePlacementTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref)
    : TaskModelFramePlacementTpl(state, id, pref, pinocchio::LOCAL,
                                 state->get_nv()) {}

template <typename Scalar>
TaskModelFramePlacementTpl<Scalar>::TaskModelFramePlacementTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const pinocchio::ReferenceFrame type, const std::size_t nu)
    : Base(state, 6, nu, true, true, false),
      id_(id),
      pref_(pref),
      oMf_inv_(pref.inverse()),
      type_(pinocchio::LOCAL),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
  set_type(type);
}

template <typename Scalar>
TaskModelFramePlacementTpl<Scalar>::TaskModelFramePlacementTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const SE3& pref, const pinocchio::ReferenceFrame type)
    : TaskModelFramePlacementTpl(state, id, pref, type, state->get_nv()) {}

template <typename Scalar>
void TaskModelFramePlacementTpl<Scalar>::calc(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rMf = oMf_inv_ * d->pinocchio->oMf[id_];
  data->y = pinocchio::log6(d->rMf).toVector();

  d->vf =
      pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_, type_);
  d->af = pinocchio::getFrameAcceleration(*pin_model_.get(), *d->pinocchio, id_,
                                          type_);
  data->v = d->vf.toVector();
  data->a = d->af.toVector();
}

template <typename Scalar>
void TaskModelFramePlacementTpl<Scalar>::calcDiff(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();

  data->Yx.setZero();
  data->Yu.setZero();
  data->Vx.setZero();
  data->Vu.setZero();
  data->Ax.setZero();
  data->Au.setZero();

  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);
  data->Yx.leftCols(nv).noalias() = d->rJf * d->fJf;

  pinocchio::getFrameAccelerationDerivatives(*pin_model_.get(), *d->pinocchio,
                                             id_, type_, d->fVdq, d->fAdq,
                                             d->fAdv, d->fVdv);
  data->Vx.leftCols(nv).noalias() = d->fVdq;
  data->Vx.rightCols(nv).noalias() = d->fVdv;
  data->Ax.leftCols(nv).noalias() = d->fAdq;
  data->Ax.rightCols(nv).noalias() = d->fAdv;
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar> >
TaskModelFramePlacementTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
TaskModelFramePlacementTpl<NewScalar> TaskModelFramePlacementTpl<Scalar>::cast()
    const {
  typedef TaskModelFramePlacementTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, pref_.template cast<NewScalar>(), type_, nu_);
  return ret;
}

template <typename Scalar>
pinocchio::FrameIndex TaskModelFramePlacementTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename TaskModelFramePlacementTpl<Scalar>::SE3&
TaskModelFramePlacementTpl<Scalar>::get_reference() const {
  return pref_;
}

template <typename Scalar>
pinocchio::ReferenceFrame TaskModelFramePlacementTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void TaskModelFramePlacementTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void TaskModelFramePlacementTpl<Scalar>::set_reference(const SE3& reference) {
  pref_ = reference;
  oMf_inv_ = reference.inverse();
}

template <typename Scalar>
void TaskModelFramePlacementTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  switch (type) {
    case pinocchio::ReferenceFrame::LOCAL:
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      type_ = type;
      return;
    default:
      throw_pretty("Invalid argument: unsupported reference frame type");
  }
}

template <typename Scalar>
void TaskModelFramePlacementTpl<Scalar>::print(std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  typename pinocchio::SE3Tpl<Scalar>::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, pref_.rotation());
  const char* type = nullptr;
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      type = "LOCAL";
      break;
    case pinocchio::ReferenceFrame::WORLD:
      type = "WORLD";
      break;
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      type = "LOCAL_WORLD_ALIGNED";
      break;
    default:
      type = "UNKNOWN";
      break;
  }
  os << "TaskModelFramePlacement {frame=" << pin_model_->frames[id_].name
     << ", type=" << type << ", tref="
     << pref_.translation().transpose().template cast<PrintableScalar>().format(
            fmt)
     << ", qref="
     << qref.coeffs().transpose().template cast<PrintableScalar>().format(fmt)
     << "}";
}

}  // namespace crocoddyl
