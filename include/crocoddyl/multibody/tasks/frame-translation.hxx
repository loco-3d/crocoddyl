///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelFrameTranslationTpl<Scalar>::TaskModelFrameTranslationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref, const std::size_t nu)
    : TaskModelFrameTranslationTpl(state, id, xref, pinocchio::LOCAL, nu) {}

template <typename Scalar>
TaskModelFrameTranslationTpl<Scalar>::TaskModelFrameTranslationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref, const pinocchio::ReferenceFrame type,
    const std::size_t nu)
    : Base(state, 3, nu, true, true, true),
      id_(id),
      xref_(xref),
      type_(pinocchio::WORLD),
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
TaskModelFrameTranslationTpl<Scalar>::TaskModelFrameTranslationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref)
    : TaskModelFrameTranslationTpl(state, id, xref, pinocchio::LOCAL,
                                   state->get_nv()) {}

template <typename Scalar>
TaskModelFrameTranslationTpl<Scalar>::TaskModelFrameTranslationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Vector3s& xref, const pinocchio::ReferenceFrame type)
    : TaskModelFrameTranslationTpl(state, id, xref, type, state->get_nv()) {}

template <typename Scalar>
void TaskModelFrameTranslationTpl<Scalar>::calc(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  const Matrix3s& oRf = d->pinocchio->oMf[id_].rotation();
  const Vector3s dt = d->pinocchio->oMf[id_].translation() - xref_;

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      data->y.noalias() = oRf.transpose() * dt;
      d->vf = pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_,
                                          pinocchio::LOCAL);
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      // The translation task only uses the linear component, so WORLD and
      // LOCAL_WORLD_ALIGNED share the same world-velocity convention here.
      data->y = dt;
      d->vf = pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_,
                                          pinocchio::WORLD);
      break;
  }

  data->v = d->vf.linear();
  if (data->compute_acceleration) {
    d->af = pinocchio::getFrameAcceleration(
        *pin_model_.get(), *d->pinocchio, id_,
        type_ == pinocchio::LOCAL ? pinocchio::LOCAL : pinocchio::WORLD);
    data->a = d->af.linear();
  } else {
    d->af = Motion::Zero();
    data->a.setZero();
  }
}

template <typename Scalar>
void TaskModelFrameTranslationTpl<Scalar>::calcDiff(
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

  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_,
                              pinocchio::LOCAL, d->fJf);
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL: {
      data->Yx.leftCols(nv).noalias() = d->fJf.template topRows<3>();
      // The LOCAL translation error depends on the frame rotation as well as
      // the frame origin position. Pinocchio orders frame Jacobians as
      // [linear; angular], so the extra term comes from d(R^T)/dq.
      const Vector3s y = data->y.template head<3>();
      data->Yx.leftCols(nv).noalias() +=
          pinocchio::skew(y) * d->fJf.template bottomRows<3>();
      break;
    }
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      data->Yx.leftCols(nv).noalias() =
          d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
      break;
  }

  const pinocchio::ReferenceFrame deriv_type =
      (type_ == pinocchio::ReferenceFrame::LOCAL) ? pinocchio::LOCAL
                                                  : pinocchio::WORLD;
  if (!data->compute_acceleration) {
    pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio,
                                           id_, deriv_type, d->fVdq, d->fVdv);
    data->Vx.leftCols(nv).noalias() = d->fVdq.template topRows<3>();
    data->Vx.rightCols(nv).noalias() = d->fVdv.template topRows<3>();
    d->fAdq.setZero();
    d->fAdv.setZero();
    return;
  }

  pinocchio::getFrameAccelerationDerivatives(*pin_model_.get(), *d->pinocchio,
                                             id_, deriv_type, d->fVdq, d->fAdq,
                                             d->fAdv, d->fVdv);
  data->Vx.leftCols(nv).noalias() = d->fVdq.template topRows<3>();
  data->Vx.rightCols(nv).noalias() = d->fVdv.template topRows<3>();
  data->Ax.leftCols(nv).noalias() = d->fAdq.template topRows<3>();
  data->Ax.rightCols(nv).noalias() = d->fAdv.template topRows<3>();
  if (d->joint != nullptr) {
    data->Ax.noalias() += d->fVdv.template topRows<3>() * d->joint->da_dx;
    data->Au.noalias() = d->fVdv.template topRows<3>() * d->joint->da_du;
  }
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar>>
TaskModelFrameTranslationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
TaskModelFrameTranslationTpl<NewScalar>
TaskModelFrameTranslationTpl<Scalar>::cast() const {
  typedef TaskModelFrameTranslationTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, xref_.template cast<NewScalar>(), type_, nu_);
  return ret;
}

template <typename Scalar>
pinocchio::FrameIndex TaskModelFrameTranslationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
TaskModelFrameTranslationTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
pinocchio::ReferenceFrame TaskModelFrameTranslationTpl<Scalar>::get_type()
    const {
  return type_;
}

template <typename Scalar>
void TaskModelFrameTranslationTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void TaskModelFrameTranslationTpl<Scalar>::set_reference(
    const Vector3s& reference) {
  xref_ = reference;
}

template <typename Scalar>
void TaskModelFrameTranslationTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  switch (type) {
    case pinocchio::ReferenceFrame::LOCAL:
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      type_ = type;
      return;
  }
}

template <typename Scalar>
void TaskModelFrameTranslationTpl<Scalar>::print(std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
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
  }
  os << "TaskModelFrameTranslation {frame=" << pin_model_->frames[id_].name
     << ", type=" << type << ", tref="
     << xref_.transpose().template cast<PrintableScalar>().format(fmt) << "}";
}

}  // namespace crocoddyl
